#!/bin/bash

# 默认路径
DEFAULT_PATH="./DataStorage/"

# 定义路径
SAVE_PATH="$DEFAULT_PATH/test_result"
RGB_CRF_PATH="$DEFAULT_PATH/rgb_crf_result"
SAL_CRF_PATH="$DEFAULT_PATH/sal_crf_result"
CRF_PATH="$DEFAULT_PATH/final_result"

# 创建文件夹
mkdir -p "$SAVE_PATH"
mkdir -p "$RGB_CRF_PATH"
mkdir -p "$SAL_CRF_PATH"
mkdir -p "$CRF_PATH"

# 执行命令
python train.py
python test.py --model_path="./DataStorage/model-best"

python utils/denseCRF.py --sm_dir "$SAVE_PATH" --crf_dir "$RGB_CRF_PATH" --input_dir "Data/color/test"
python utils/denseCRF.py --sm_dir "$SAVE_PATH" --crf_dir "$SAL_CRF_PATH" --input_dir "Data/spec_sal/test"
python utils/CRF_add.py --rgb_crf_dir "$RGB_CRF_PATH" --sal_crf_dir "$SAL_CRF_PATH" --out_dir "$CRF_PATH"
python eval/Eval_MulMeth_MulDataset.py --sm_dir "$CRF_PATH"
python eval/evaluate.py --sm_dir "$CRF_PATH"
