import os

import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange

import dataset_HSI as Dataset
from lscloss import *
from models.modeling import VisionTransformer, get_config
from utils import AverageMeter, set_seed, clip_gradient, mean_square_error

save_path = "./DataStorage"
data_path = "./Data"

loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 5, "rgb": 0.03}]
loss_lsc_kernels_desc_defaults_spec = [{"weight": 1, "xy": 0.003, "rgb": 3}]
loss_lsc_radius = 5


def setup(config):
    model = VisionTransformer(config, config.img_size, zero_head=False)
    model.load_from(np.load(config.pretrained_dir))
    return model


def valid(test_loader, net):
    maes = AverageMeter()
    net.train(False)
    with torch.no_grad():
        for image, gt, spec, (H, W), name in test_loader:
            image, shape = image.cuda().float(), (H, W)
            spec = spec.cuda().float()
            spec = spec.repeat([1, 3, 1, 1])
            gt = gt.cuda().float()
            image = F.interpolate(image, size=352, mode="bilinear", align_corners=True)

            out, _, _, _ = net(image, spec)
            pred = torch.sigmoid(out)
            pred = F.interpolate(pred, (H[0], W[0]), mode="bilinear", align_corners=True)

            head = "./DataStorage/valid_result/"
            if not os.path.exists(head):
                os.makedirs(head)
            for i in range(pred.shape[0]):
                cv2.imwrite(
                    head + "/" + name[i] + "_out.jpg", pred[i, 0].cpu().numpy() * 255
                )
                cv2.imwrite(head + "/" + name[i] + "_gt.jpg", gt[i, 0].cpu().numpy() * 255)

            mae = mean_square_error(gt, pred)
            maes.update(mae)
    return maes.avg


def train(cfg, loader, test_loader, net):
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True, )
    mae_loss_record = 1

    CE = torch.nn.BCELoss(reduction="none").cuda()
    CE_mean = torch.nn.BCELoss().cuda()
    loss_lsc = LocalSaliencyCoherence().cuda()

    for epoch in trange(cfg.epoch):
        losses = AverageMeter()
        net.train(True)

        optimizer.param_groups[0]["lr"] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr

        for step, (image, spec_sal, gt, mask, edge_gt, RGB_image) in enumerate(loader):
            image, spec_sal, gt, mask, edge_gt, RGB_image = (
                image.type(torch.FloatTensor).cuda(),
                spec_sal.type(torch.FloatTensor).cuda(),
                gt.type(torch.FloatTensor).cuda(),
                mask.type(torch.FloatTensor).cuda(),
                edge_gt.type(torch.FloatTensor).cuda(),
                RGB_image.type(torch.FloatTensor).cuda(),
            )

            out_final, out_edg, refine_map, vis = net(image, spec_sal)

            refine_loss = CE_mean(refine_map,
                                  F.interpolate(gt, refine_map.shape[2:], mode="bilinear", align_corners=False))
            out_final_prob = torch.sigmoid(out_final)

            img_size = image.size(2) * image.size(3) * image.size(0)
            ratio = img_size / torch.sum(mask)
            sal_loss2 = ratio * CE(out_final_prob * mask, gt * mask)

            out_final_prob = F.interpolate(
                out_final_prob, scale_factor=0.25, mode="bilinear", align_corners=False
            )
            image_ = F.interpolate(
                RGB_image, scale_factor=0.25, mode="bilinear", align_corners=False
            )
            sample = {"rgb": image_}
            sal_ = F.interpolate(
                spec_sal, scale_factor=0.25, mode="bilinear", align_corners=False
            )
            sample_sal = {"rgb": sal_}

            # after sigmoid
            loss2_lsc_rgb = loss_lsc(
                out_final_prob,
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                image_.shape[2],
                image_.shape[3],
                out_kernels_vis=False,
            )
            loss2_lsc_spec = loss_lsc(
                out_final_prob,
                loss_lsc_kernels_desc_defaults_spec,
                loss_lsc_radius,
                sample_sal,
                image_.shape[2],
                image_.shape[3],
                out_kernels_vis=False,
            )
            loss2_lsc = loss2_lsc_spec["loss"].mean() + loss2_lsc_rgb["loss"].mean()
            edge_loss = 1.0 * CE(torch.sigmoid(out_edg), edge_gt)

            loss = torch.mean(edge_loss) + torch.mean(sal_loss2) + loss2_lsc + refine_loss

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, cfg.lr)
            optimizer.step()
            losses.update(loss)

        mae_loss = valid(test_loader, net)

        if mae_loss < mae_loss_record:
            if not os.path.exists(cfg.savepath):
                os.makedirs(cfg.savepath)
            torch.save(net.state_dict(), cfg.savepath + "/model-best")
            mae_loss_record = mae_loss


if __name__ == "__main__":
    set_seed(7)
    config = get_config()
    net = setup(config).cuda()
    train_cfg = Dataset.Config(
        datapath=data_path,
        savepath=save_path,
        mode="train",
        batch=6,
        lr=5e-3,
        momen=0.9,
        decay=5e-4,
        epoch=200,
    )
    train_data = Dataset.Data(train_cfg)
    train_loader = DataLoader(
        train_data,
        collate_fn=train_data.collate,
        batch_size=train_cfg.batch,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    test_cfg = Dataset.Config(
        datapath=data_path,
        mode="test",
    )
    test_data = Dataset.Data(test_cfg)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4)

    train(train_cfg, train_loader, test_loader, net)
