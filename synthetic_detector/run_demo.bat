@echo off
REM 合成图像检测器演示脚本

set PYTHONPATH=%PYTHONPATH%;%~dp0\..

python demo/demo.py ^
    --input "%~dp0\..\data\demo_images" ^
    --checkpoint checkpoints/detector_best.pth ^
    --output_dir demo_results ^
    --img_size 256 ^
    --threshold 0.5 ^
    --visualize ^
    --device cuda
