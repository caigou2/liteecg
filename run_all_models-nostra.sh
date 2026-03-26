#!/bin/bash

# 设置编码（Linux 默认通常为 UTF-8，此行可选）
export LANG=en_US.UTF-8

echo "========================================"

# 请根据您的 Linux 环境修改 Anaconda 环境的 Python 路径
# 通常路径类似于: /home/用户名/anaconda3/envs/hcyenv/bin/python
PYTHON_PATH="/home/source/wangyang/miniconda3/envs/LiteECG_env/bin/python"

EPOCHS=30
BATCH_SIZE=256
LEARNING_RATE="3e-3"

# 定义执行训练的函数，减少重复代码
run_train() {
    MODEL_NAME=$1
    echo "========================================"
    echo "$MODEL_NAME"
    echo "========================================"
    $PYTHON_PATH trainer.py \
        --model "$MODEL_NAME" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --cache \
        --no-stratified
}

# 按顺序执行各个模型
run_train "ecgnet"
run_train "se_ecgnet"
run_train "bircnn"
run_train "resnet"
run_train "deepecgnet"
run_train "ldcnn"
run_train "liteecgnet"

echo "========================================"
read -p "Press [Enter] key to continue..."