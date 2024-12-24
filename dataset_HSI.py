import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import h5py
import torch.nn.functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, spec_sal, gt, mask, edge_gt, RGB_image):
        for t in self.transforms:
            img, spec_sal, gt, mask, edge_gt, RGB_image = t(
                img, spec_sal, gt, mask, edge_gt, RGB_image
            )
        return img, spec_sal, gt, mask, edge_gt, RGB_image


class RandomHorizontallyFlip(object):
    def __call__(self, img, spec_sal, gt, mask, edge_gt, RGB_image):
        if np.random.random() < 0.5:
            return (
                img[:, :, torch.arange(img.shape[2] - 1, -1, -1)],
                spec_sal.transpose(Image.FLIP_LEFT_RIGHT),
                gt.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
                edge_gt.transpose(Image.FLIP_LEFT_RIGHT),
                RGB_image.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, spec_sal, gt, mask, edge_gt, RGB_image


class RandomCrop(object):
    def __call__(self, image, spec_sal, gt, mask, edge_gt, RGB_image):
        image = np.array(image)
        spec_sal = np.array(spec_sal)
        spec_sal = np.stack([spec_sal, spec_sal, spec_sal], axis=2)
        gt = np.array(gt)
        mask = np.array(mask)
        edge_gt = np.array(edge_gt)
        RGB_image = np.array(RGB_image)
        H, W = gt.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        image = image[:, p0:p1, p2:p3]
        spec_sal = Image.fromarray(spec_sal[p0:p1, p2:p3])
        gt = Image.fromarray(gt[p0:p1, p2:p3].astype("uint8"))
        mask = Image.fromarray(mask[p0:p1, p2:p3].astype("uint8"))
        edge_gt = Image.fromarray(edge_gt[p0:p1, p2:p3].astype("uint8"))
        RGB_image = Image.fromarray(RGB_image[p0:p1, p2:p3].astype("uint8"))
        return image, spec_sal, gt, mask, edge_gt, RGB_image


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.joint_transform_train = Compose(
            [
                RandomHorizontallyFlip(),
                RandomCrop(),
            ]
        )
        self.rgb_transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4042092044546734,0.47479873506817966,0.4320108453568537],
                    [0.1981914442818379,0.19832119061320555,0.1936594745988259],
                ),
            ]
        )
        self.mask_transform_train = transforms.ToTensor()

        self.gt_transform_test = transforms.ToTensor()
        self.spec_sal_transform_test = transforms.Compose(
            [transforms.Resize((352, 352)), transforms.ToTensor()]
        )

        if cfg.mode == "train":
            with open(cfg.datapath + "/train.txt", "r") as lines:
                self.samples = []
                for line in lines:
                    self.samples.append(line.strip())
            self.gts = []
            self.edge_gts = []
            self.masks = []
            self.RGB_images = []
        elif cfg.mode == "test":
            with open(cfg.datapath + "/test.txt", "r") as lines:
                self.samples = []
                for line in lines:
                    self.samples.append(line.strip())
            self.gts = []

        self.images = []
        self.spec_sals = []
        for name in tqdm(self.samples):
            name = name.split(".")[0]
            mat = h5py.File(self.cfg.datapath + "/hyperspectral/" + cfg.mode + "/" + name + ".mat", "r")
            image = np.float32(np.array(mat["hypercube"]))  # (C,H,W)
            image = image[:, :, torch.arange(image.shape[2] - 1, -1, -1)]
            image = torch.from_numpy(image / np.max(image))  # (C,H,W)
            self.images.append(image)

            if self.cfg.mode == "train":
                gt = Image.open(
                    self.cfg.datapath + "/training/pseudo-label/" + name + ".jpg"
                ).convert("L")
                mask = Image.open(
                    self.cfg.datapath + "/training/mask/" + name + ".jpg"
                ).convert("L")
                edge_gt = Image.open(
                    self.cfg.datapath + "/training/edge_gt/" + name + ".jpg"
                ).convert("L")
                RGB_image = Image.open(
                    self.cfg.datapath + "/color/train/" + name + ".jpg"
                ).convert("RGB")
                self.gts.append(gt)
                self.edge_gts.append(edge_gt)
                self.masks.append(mask)
                self.RGB_images.append(RGB_image)
            elif self.cfg.mode == "test":
                gt = Image.open(
                    self.cfg.datapath + "/GT/test/" + name + ".jpg"
                ).convert("L")
                self.gts.append(gt)

            spec_sal = Image.open(self.cfg.datapath + "/spec_sal/" + self.cfg.mode + "/" + name + ".jpg").convert(
                "L"
            )
            self.spec_sals.append(spec_sal)
        print(f"{len(self.samples)} data loaded!")

    def __getitem__(self, idx):
        name = self.samples[idx]
        name = name.split(".")[0]
        image = self.images[idx]
        spec_sal = self.spec_sals[idx]

        if self.cfg.mode == "train":
            gt = self.gts[idx]
            mask = self.masks[idx]
            edge_gt = self.edge_gts[idx]
            RGB_image = self.RGB_images[idx]
            image, spec_sal, gt, mask, edge_gt, RGB_image = self.joint_transform_train(
                image, spec_sal, gt, mask, edge_gt, RGB_image
            )
            image = torch.from_numpy(image)
            spec_sal = self.mask_transform_train(spec_sal)
            gt = self.mask_transform_train(gt)
            mask = self.mask_transform_train(mask)
            edge_gt = self.mask_transform_train(edge_gt)
            RGB_image = self.rgb_transform_train(RGB_image)
            return image, spec_sal, gt, mask, edge_gt, RGB_image
        else:
            gt = self.gts[idx]
            shape = gt.size[::-1]
            gt = self.gt_transform_test(gt)
            spec_sal = self.spec_sal_transform_test(spec_sal)
            return image, gt, spec_sal, shape, name

    def __len__(self):
        return len(self.gts)

    def collate(self, batch):
        size = 352
        image, spec_sal, gt, mask, edge_gt, RGB_image = [
            list(item) for item in zip(*batch)
        ]  # [C,H,W]
        for i in range(len(batch)):
            spec_sal[i] = np.array(spec_sal[i]).transpose((1, 2, 0))
            gt[i] = np.array(gt[i]).transpose((1, 2, 0))
            mask[i] = np.array(mask[i]).transpose((1, 2, 0))
            edge_gt[i] = np.array(edge_gt[i]).transpose((1, 2, 0))
            RGB_image[i] = np.array(RGB_image[i]).transpose((1, 2, 0))
            
            image[i] = F.interpolate(image[i].unsqueeze(0), size=(size, size), mode="bilinear",
                                     align_corners=False).squeeze(0)
            spec_sal[i] = cv2.resize(
                spec_sal[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )
            gt[i] = cv2.resize(
                gt[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )
            mask[i] = cv2.resize(
                mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )
            edge_gt[i] = cv2.resize(
                edge_gt[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )
            RGB_image[i] = cv2.resize(
                RGB_image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR
            )

        image = torch.stack(image)
        spec_sal = torch.from_numpy(np.stack(spec_sal, axis=0)).permute(0, 3, 1, 2)
        gt = torch.from_numpy(np.stack(gt, axis=0)).unsqueeze(dim=1)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(dim=1)
        edge_gt = torch.from_numpy(np.stack(edge_gt, axis=0)).unsqueeze(dim=1)
        RGB_image = torch.from_numpy(np.stack(RGB_image, axis=0)).permute(0, 3, 1, 2)
        return image, spec_sal, gt, mask, edge_gt, RGB_image
