import os

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sm_dir', type=str)
parser.add_argument('--crf_dir', type=str)
parser.add_argument('--input_dir', type=str)
args = parser.parse_args()


# set path
input_path = args.input_dir
sal_path = args.sm_dir
output_path = args.crf_dir


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


files = os.listdir(input_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)

files.sort()
for file in tqdm(files):
    # print(file)
    if (os.path.isfile(input_path + "/" + file)):
        img = cv2.imread(input_path + '/' + file, 1)

        file = file.split('.')[0] + '.jpg'
        annos = cv2.imread(sal_path + '/' + file, 0)
        labels = relabel_sequential(cv2.imread(sal_path + '/' + file, 0))[0].flatten()
        output = output_path + '/' + file

        EPSILON = 1e-8

        M = 2  # salient or not
        tau = 1.05

        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

        anno_norm = annos / 255.
        n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

        U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

        # Do the inference
        infer = np.array(d.inference(10)).astype('float32')  # number of the inferences
        res = infer[1, :]

        res = res * 255
        res = res.reshape(img.shape[:2])
        # _, res = cv2.threshold(res, 130, 255, cv2.THRESH_BINARY)
        cv2.imwrite(output, res.astype('uint8'))
