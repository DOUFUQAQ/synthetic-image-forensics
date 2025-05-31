@echo off
REM 合成图像检测器评估脚本

set PYTHONPATH=%PYTHONPATH%;%~dp0\..

python evaluate/evaluate_detector.py ^
    --data_dir "%~dp0\..\data\DeepFaceGen" ^
    --checkpoint checkpoints/detector_best.pth ^
    --subset_ratio 0.2 ^
    --img_size 256 ^
    --batch_size 32 ^
    --output_dir results ^
    --visualize ^
    --device cuda
