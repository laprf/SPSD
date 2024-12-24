import argparse
import os
import time

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataset_HSI as dataset
from models.modeling import VisionTransformer, get_config

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="DataStorage/model-best")
args = parser.parse_args()

model_path = args.model_path
save_path = "DataStorage/test_result"


class Test(object):
    def __init__(self, Dataset, Network, Path, snapshot):
        ## dataset
        self.cfg = Dataset.Config(datapath=Path, snapshot=snapshot, mode="test")

        config = get_config()
        self.net = Network(config, img_size=352)
        self.net.cuda()

        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(
            self.cfg.snapshot, map_location=torch.device("cpu")
        )
        pretrained_dict = {
            k.replace("module.", ""): v
            for k, v in pretrained_dict.items()
            if (k.replace("module.", "") in model_dict)
        }

        # check unloaded weights
        for k, v in model_dict.items():
            if k in pretrained_dict.keys():
                pass
            else:
                print("miss keys in pretrained_dict: {}".format(k))

        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

        self.net.train(False)

        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)

    def save(self):
        with torch.no_grad():
            total_time = []
            for image, gt, spec, (H, W), name in self.loader:
                image, shape = image.cuda().float(), (H, W)
                image = F.interpolate(image, size=(352, 352), mode="bilinear", align_corners=True)
                spec = spec.cuda().float()
                spec = spec.repeat([1, 3, 1, 1])
                gt = gt.cuda().float()

                # out, refine_map, e1, e2 = net(image, spec_sal)
                start = time.time()
                out, _, _, vis = self.net(image, spec)
                end = time.time()
                total_time.append(end - start)
                pred = torch.sigmoid(out)
                pred = F.interpolate(pred, (H[0], W[0]), mode="bilinear", align_corners=True)

                head = save_path
                for i in range(pred.shape[0]):
                    cv2.imwrite(
                        head + "/" + name[i].split(".")[0] + ".jpg", pred[i, 0].cpu().numpy() * 255
                    )
            print("average fps: ", len(total_time) / sum(total_time))


if __name__ == "__main__":
    t = Test(
        dataset,
        VisionTransformer,
        "./Data",
        model_path,
    )
    t.save()
