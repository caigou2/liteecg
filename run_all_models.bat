@echo off
chcp 65001 >nul
echo ========================================

set PYTHON_PATH=C:/Users/Backup/anaconda3/envs/hcyenv/python.exe

set EPOCHS=30
set BATCH_SIZE=256
set LEARNING_RATE=3e-3

echo ========================================
echo ecgnet
echo ========================================
%PYTHON_PATH% trainer.py --model ecgnet --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --cache

echo ========================================
echo se_ecgnet
echo ========================================
%PYTHON_PATH% trainer.py --model se_ecgnet --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --cache
echo ========================================
echo bircnn
echo ========================================
%PYTHON_PATH% trainer.py --model bircnn --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --cache
echo ========================================
echo resnet
echo ========================================
%PYTHON_PATH% trainer.py --model resnet --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --cache

echo ========================================
echo deepecgnet
echo ========================================
%PYTHON_PATH% trainer.py --model deepecgnet --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --cache


echo ========================================
echo ldcnn
echo ========================================
%PYTHON_PATH% trainer.py --model ldcnn --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --cache

echo ========================================
echo liteecgnet
echo ========================================
%PYTHON_PATH% trainer.py --model liteecgnet --epochs %EPOCHS% --batch-size %BATCH_SIZE% --lr %LEARNING_RATE% --cache
pause
