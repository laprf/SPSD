import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rgb_crf_dir', type=str)
parser.add_argument('--sal_crf_dir', type=str)
parser.add_argument('--out_dir', type=str)
args = parser.parse_args()

rgb_CRF_path = args.rgb_crf_dir
sal_path = args.sal_crf_dir
output_path = args.out_dir

files = os.listdir(rgb_CRF_path)
files.sort()

#读取rgb图像的CRF结果和sal图像的CRF结果，然后进行与运算
for file in files:
    rgb_CRF = cv2.imread(rgb_CRF_path + '/' + file, 0)
    sal_CRF = cv2.imread(sal_path + '/' + file, 0)
    and_CRF = cv2.bitwise_and(rgb_CRF, sal_CRF)
    cv2.imwrite(output_path + '/' + file, and_CRF)