@echo off
REM 合成图像检测器训练脚本

set PYTHONPATH=%PYTHONPATH%;%~dp0\..

python train/train_detector.py ^
    --data_dir "%~dp0\..\data\DeepFaceGen" ^
    --subset_ratio 0.1 ^
    --max_samples 5000 ^
    --img_size 256 ^
    --batch_size 32 ^
    --epochs 50 ^
    --lr 0.001 ^
    --weight_decay 1e-4 ^
    --pretrained ^
    --save_dir checkpoints ^
    --log_dir logs ^
    --device cuda
