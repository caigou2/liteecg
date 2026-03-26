#!/bin/bash

# ========================================
# 1. 环境配置
# ========================================

# 注意：请根据您 Linux 服务器上的实际路径修改 Python 位置
# 通常在 anaconda 的 bin 文件夹下，例如：/home/yourname/anaconda3/envs/hcyenv/bin/python
PYTHON_PATH="/home/source/wangyang/miniconda3/envs/LiteECG_env/bin/python"

# 如果已经在环境中，也可以直接写 python
# PYTHON_PATH="python"

EPOCHS=30
BATCH_SIZE=256
LEARNING_RATE=3e-3

# ========================================
# 2. 训练流程
# ========================================

# 定义要运行的模型列表
MODELS=("ecgnet" "se_ecgnet" "bircnn" "resnet" "deepecgnet" "ldcnn" "liteecgnet")

for MODEL in "${MODELS[@]}"
do
    echo "========================================"
    echo "正在训练: $MODEL"
    echo "========================================"
    
    $PYTHON_PATH trainer.py \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --cache
        
    echo -e "\n"
done

echo "========================================"
echo "所有模型训练任务已完成！"
echo "========================================"

# 模拟 Windows 的 pause 功能
read -n 1 -s -r -p "按任意键退出..."