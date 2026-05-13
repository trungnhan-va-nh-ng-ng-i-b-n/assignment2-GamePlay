@echo off
:: ============================================================
:: server_setup.bat — Setup môi trường và chạy benchmark
:: Chạy lần đầu trên Windows server
:: ============================================================

echo ========================================
echo  Step 1: Clone / Pull repo
echo ========================================
if exist ".git" (
    echo [INFO] Repo da co, pulling VP/ml branch...
    git fetch origin
    git checkout VP/ml
    git pull origin VP/ml
) else (
    echo [INFO] Clone repo...
    git clone -b VP/ml https://github.com/trungnhan-va-nh-ng-ng-i-b-n/assignment2-GamePlay.git .
)

echo.
echo ========================================
echo  Step 2: Cai dat Python dependencies
echo ========================================
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pygame

echo.
echo ========================================
echo  Step 3: Kiem tra model
echo ========================================
if not exist "models\best_mlp_model (7).pt" (
    echo [WARNING] Chua co model file!
    echo [WARNING] Copy file "best_mlp_model (7).pt" vao thu muc models\
    echo [WARNING] Sau do chay lai server_run.bat
    pause
    exit /b 1
) else (
    echo [OK] Model file found!
)

echo.
echo ========================================
echo  Setup DONE! Chay server_run.bat de benchmark
echo ========================================
pause
