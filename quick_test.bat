@echo off
chcp 65001 >nul
echo quick test all models （1 epoch）
echo ========================================

set PYTHON_PATH=C:/Users/Backup/anaconda3/envs/hcyenv/python.exe

echo test liteecgnet...
%PYTHON_PATH% trainer.py --model liteecgnet --epochs 1 --batch-size 256 --lr 3e-3 --cache

echo test ecgnet...
%PYTHON_PATH% trainer.py --model ecgnet --epochs 1 --batch-size 256 --lr 3e-3 --cache

echo test se_ecgnet...
%PYTHON_PATH% trainer.py --model se_ecgnet --epochs 1 --batch-size 256 --lr 3e-3 --cache

echo test resnet...
%PYTHON_PATH% trainer.py --model resnet --epochs 1 --batch-size 256 --lr 3e-3 --cache


echo test deepecgnet...
%PYTHON_PATH% trainer.py --model deepecgnet --epochs 1 --batch-size 256 --lr 3e-3 --cache

echo test ldcnn...
%PYTHON_PATH% trainer.py --model ldcnn --epochs 1 --batch-size 256 --lr 3e-3 --cache
echo test bircnn...
%PYTHON_PATH% trainer.py --model bircnn --epochs 1 --batch-size 256 --lr 3e-3 --cache

echo ========================================
echo test complete!
pause
