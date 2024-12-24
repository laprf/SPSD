import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sm_dir", type=str, default=None)
args = parser.parse_args()

import warnings

warnings.filterwarnings('ignore')


class Evaluate(object):

    def __init__(self, dir_GT):
        self.dir_GT = dir_GT
        self.image_list = os.listdir(dir_GT)
        self.image_list.sort()
        self.num = len(self.image_list)
        self.colors = ['red', 'blue', 'green', 'orange', 'black', 'brown', 'gray']

    def PR(self, dir_sal_list, dir_output, curve_name="PRcurve.jpg", show_fig=False):
        print("<------ PR Curve ------>")
        fig = plt.figure(figsize=(6, 6))

        for j, dir_sal in enumerate(dir_sal_list):
            prec_all = np.zeros((self.num, 256))
            rec_all = np.zeros((self.num, 256))
            F_all = np.zeros((self.num))

            for i, GT_name in enumerate(self.image_list):
                img_GT = self.__read_GT(self.dir_GT, GT_name)
                sal_name = os.path.splitext(GT_name)[0] + ".jpg"
                img_sal = self.__read_image(dir_sal, sal_name)

                prec, rec, f = self.__cal_PR(img_GT, img_sal)
                prec_all[i, :] = prec
                rec_all[i, :] = rec
                F_all[i] = f

            PREC = np.mean(prec_all, axis=0)
            REC = np.mean(rec_all, axis=0)
            F = np.mean(F_all)

            label = dir_sal.split('/')[-2]
            plt.plot(REC, PREC, c=self.colors[j], label=label)
            print("[%d] %s:  Prec: %.4f, Rec: %.4f, F: %.4f" %
                  (j + 1, label, PREC[128], REC[128], F))

        plt.legend(loc='upper right')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR curve')
        plt.axis([0, 1, 0, 1])
        plt.grid(ls='--')
        plt.savefig(os.path.join(dir_output, curve_name))
        if show_fig:
            plt.show()

    def ROC(self, dir_sal_list, dir_output, curve_name="ROCcurve.jpg", show_fig=False):
        print("<------ ROC Curve ------>")
        fig = plt.figure(figsize=(6, 6))

        for j, dir_sal in enumerate(dir_sal_list):
            TPR_all = np.zeros((self.num, 256))
            FPR_all = np.zeros((self.num, 256))

            for i, GT_name in enumerate(self.image_list):
                img_GT = self.__read_GT(self.dir_GT, GT_name)
                sal_name = os.path.splitext(GT_name)[0] + ".jpg"
                img_sal = self.__read_image(dir_sal, sal_name)

                tpr, fpr = self.__cal_ROC(img_GT, img_sal)
                TPR_all[i, :] = tpr
                FPR_all[i, :] = fpr

            TPR = np.mean(TPR_all, axis=0)
            FPR = np.mean(FPR_all, axis=0)
            AUC = self.__cal_AUC(TPR, FPR)

            label = dir_sal.split('/')[-2]
            plt.plot(FPR, TPR, c=self.colors[j], label=label)
            print("[%d] %s:  AUC: %.4f" % (j + 1, label, AUC))

        plt.legend(loc='upper right')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.axis([0, 1, 0, 1])
        plt.grid(ls='--')
        plt.savefig(os.path.join(dir_output, curve_name))
        if show_fig:
            plt.show()

    def CC(self, dir_sal_list):
        print("<------ Cross Correlation ------>")

        for j, dir_sal in enumerate(dir_sal_list):
            CC_all = np.zeros((self.num))

            for i, GT_name in enumerate(self.image_list):
                img_GT = self.__read_GT(self.dir_GT, GT_name)
                sal_name = os.path.splitext(GT_name)[0] + ".jpg"
                img_sal = self.__read_image(dir_sal, sal_name)

                cc = self.__cal_CC(img_GT, img_sal)
                CC_all[i] = cc

            CC = np.mean(CC_all)

            label = dir_sal.split('/')[-2]
            print("[%d] %s:  CC: %.4f" % (j + 1, label, CC))

    def NSS(self, dir_sal_list):
        print("<------ Normalized Scanpath Saliency ------>")

        for j, dir_sal in enumerate(dir_sal_list):
            NSS_all = np.zeros((self.num))

            for i, GT_name in enumerate(self.image_list):
                img_GT = self.__read_GT(self.dir_GT, GT_name)
                sal_name = os.path.splitext(GT_name)[0] + ".jpg"
                img_sal = self.__read_image(dir_sal, sal_name)

                nss = self.__cal_NSS(img_GT, img_sal)
                NSS_all[i] = nss

            NSS = np.mean(NSS_all)

            label = dir_sal.split('/')[-2]
            print("[%d] %s:  NSS: %.4f" % (j + 1, label, NSS))

    def __read_GT(self, dir_GT, GT_name):
        img = cv2.imread(os.path.join(dir_GT, GT_name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 128
        return img

    def __read_image(self, dir_sal, sal_name):
        img = cv2.imread(os.path.join(dir_sal, sal_name), cv2.IMREAD_GRAYSCALE)
        return img

    def __cal_PR(self, img_GT, img_sal):
        target = img_sal[img_GT]
        nontarget = img_sal[(1 - img_GT).astype(bool)]

        tp = np.zeros((256))
        fp = np.zeros((256))
        for i in range(256):
            tp[i] = np.sum(target >= i)
            fp[i] = np.sum(nontarget >= i)
        tp = np.flipud(tp)
        fp = np.flipud(fp)

        prec = tp / (tp + fp + 1e-5)
        rec = tp / target.shape[0]

        beta = 0.3
        th = 128
        f = (1 + beta) * prec[th] * rec[th] / (beta * prec[th] + rec[th] + 1e-5)
        return prec, rec, f

    def __cal_ROC(self, img_GT, img_sal):
        target = img_sal[img_GT]
        nontarget = img_sal[(1 - img_GT).astype(bool)]

        tp = np.zeros((256))
        fp = np.zeros((256))
        fn = np.zeros((256))
        tn = np.zeros((256))
        for i in range(256):
            tp[i] = np.sum(target >= i)
            fp[i] = np.sum(nontarget >= i)
            fn[i] = np.sum(target < i)
            tn[i] = np.sum(nontarget < i)
        tp = np.flipud(tp)
        fp = np.flipud(fp)
        fn = np.flipud(fn)
        tn = np.flipud(tn)

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        return tpr, fpr

    def __cal_AUC(self, TPR, FPR):
        AUC = 0
        for i in range(255):
            AUC += (TPR[i] + TPR[i + 1]) * (FPR[i + 1] - FPR[i]) / 2
        return AUC

    def __cal_CC(self, img_GT, img_sal):
        map1 = img_GT.astype(float)
        map1 = map1 - np.mean(map1)
        map2 = img_sal.astype(float) / 255
        map2 = map2 - np.mean(map2)

        cov = np.sum(map1 * map2)
        d1 = np.sum(map1 * map1)
        d2 = np.sum(map2 * map2)
        cc = cov / (np.sqrt(d1) * np.sqrt(d2) + 1e-3)
        return cc

    def __cal_NSS(self, img_GT, img_sal):
        map = img_sal.astype(float)
        map = (map - np.mean(map)) / (np.std(map) + 1e-3)

        nss = np.mean(map[img_GT])
        return nss


if __name__ == '__main__':

    dir_GT = "./Data/GT/test"
    dir_sal_list = [args.sm_dir]
    dir_output = "./curve/"
    mode = {"PR": False,
            "ROC": True,
            "CC": True,
            "NSS": False}

    eva = Evaluate(dir_GT)
    if mode["PR"]:
        eva.PR(dir_sal_list, dir_output)
    if mode["ROC"]:
        eva.ROC(dir_sal_list, dir_output)
    if mode["CC"]:
        eva.CC(dir_sal_list)
    if mode["NSS"]:
        eva.NSS(dir_sal_list)
