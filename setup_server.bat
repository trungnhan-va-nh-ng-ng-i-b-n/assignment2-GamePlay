@echo off
REM ================================================================
REM setup_server.bat — Cài Python + deps trên Windows server
REM Chạy file này trên server sau khi copy project lên
REM ================================================================

echo === Checking Python ===
python --version 2>nul
if errorlevel 1 (
    echo Python not found! Download from https://python.org
    echo Sau khi cai xong chay lai file nay.
    pause
    exit /b 1
)

echo === Installing dependencies ===
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
pip install pyarrow

echo === Done! Now run data generation: ===
echo python -u generate_parallel.py --games 20000 --level 7 --think 1.0 --workers 80 --output new_L7_20k.npz
pause
