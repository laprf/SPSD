import argparse
import os

from saliency_toolbox import calculate_measures

parser = argparse.ArgumentParser()
parser.add_argument("--sm_dir", type=str, default=None)
args = parser.parse_args()

# 将文件写入excel表格和txt中
record = "result"
logfile = record + ".txt"

res = {}
for method in [
    "SPSD"
]:
    for datasetname in ["HSOD-BIT"]:
        gt_dir = "./Data/GT/test"
        sm_dir = args.sm_dir

        if not os.path.exists(sm_dir):
            res[datasetname] = {"Max-F": 0, "Mean-F": 0, "S-measure": 0, "MAE": 0}
            continue
        print("Evaluate " + method + " on " + datasetname + " dataset: ")

        res[datasetname] = calculate_measures(
            gt_dir,
            sm_dir,
            [
                "MAE",
                "E-measure",
                # "S-measure",
                # "Max-F",
                "Adp-F",
                # "Wgt-F",
            ],  # 'MAE', 'Adp-E-measure', 'S-measure', 'Max-F', 'Mean-F', 'Adp-F', 'Wgt-F'
            save="./eval/save",
        )

        with open(logfile, "a") as f:  # 'a' 打开文件接着写
            f.write("\n------------cut off line--------------\n")
            f.write(
                "{} dataset with {} method get {:.4f} mae, {:.4f} e-measure, {:.4f} adp-f \n".format(
                    datasetname,
                    method,
                    # res[datasetname]["Precision"].mean(),
                    # res[datasetname]["Recall"].mean(),
                    res[datasetname]["MAE"],
                    res[datasetname]["E-measure"],
                    # res[datasetname]["S-measure"],
                    res[datasetname]["Adp-F"],
                )
            )
