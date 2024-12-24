# Spectrum-oriented Point-supervised Saliency Detector for Hyperspectral Images [TIM 2024]

by [Peifu Liu](https://scholar.google.com/citations?user=yrRXe-8AAAAJ), [Tingfa Xu](https://scholar.google.com/citations?user=vmDc8dwAAAAJ), Guokai Shi, Jingxuan Xu, [Huan Chen](https://scholar.google.com/citations?user=1G6Mj24AAAAJ), and [Jianan Li](https://scholar.google.com/citations?user=sQ_nP0ZaMn0C).

## Requirements
```bash
conda create --name SPSD --file requirements.txt
```
pydensecrf should be installed by:
```bash
pip install cython
conda install -c conda-forge pydensecrf
```

## Getting Started
### Prepare Data
Please download from [Baidu Netdisk](https://pan.baidu.com/s/1WJYw0xffQAkWubN5AOuvhA?pwd=jgi0) and place them in the `Data` folder. The folder structure should be as follows:
```
/Data
    /color
        /train
        /test
    /GT
        /train
        /test
    /hyperspectral
        /train
        /test
    /spec_sal
        /train
        /test
    /training
        /edge_gt
        /mask
        /pseudo-label
    train.txt
    test.txt
```
Note: The [hyperspectral data](https://pan.baidu.com/s/1xN9wpC5RiPiNFFSweWd75A?pwd=jhaj) we use is obtained by downsampling the raw data. For the original data, please refer to [HSOD-BIT](https://github.com/anonymous0519/HSOD-BIT).

## Training, Testing, and Evaluation
Just run the following command:
```bash
bash run.sh
```
We also provide our trained model and detection results. Please download from [Baidu Netdisk](https://pan.baidu.com/s/1WJYw0xffQAkWubN5AOuvhA?pwd=jgi0) for reproduction.  

## Acknowledgement
We refer to the following repositories:
- [PSOD](https://github.com/shuyonggao/PSOD)
- [HSOD-BIT](https://github.com/anonymous0519/HSOD-BIT)
- [Evaluate-SOD](https://github.com/Hanqer/Evaluate-SOD)

Thanks for their great work!


## License
This project is licensed under the [LICENSE.md](LICENSE.md).