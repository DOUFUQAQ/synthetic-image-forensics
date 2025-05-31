@echo off
REM DeepFaceGen数据集预处理脚本

set PYTHONPATH=%PYTHONPATH%;%~dp0\..

python data/prepare_deepfacegen.py ^
    --src_dir "%~dp0\..\data\DeepFaceGen_original" ^
    --dst_dir "%~dp0\..\data\DeepFaceGen_processed" ^
    --ratio 0.1 ^
    --max_samples 5000 ^
    --img_size 256 ^
    --resize ^
    --num_workers 4
