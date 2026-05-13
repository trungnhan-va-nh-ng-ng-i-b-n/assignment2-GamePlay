@echo off
echo ============================================
echo  Setup New Machine for Data Generation
echo ============================================

echo.
echo [1/4] Installing Python 3.11...
winget install Python.Python.3.11 -e --silent
echo Done. Please restart terminal if this is first install.

echo.
echo [2/4] Installing git...
winget install Git.Git -e --silent

echo.
echo [3/4] Installing Python packages...
py -m pip install --upgrade pip
py -m pip install numpy
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo.
echo [4/4] Cloning repo...
git clone https://github.com/trungnhan-va-nh-ng-ng-i-b-n/assignment2-GamePlay.git
cd assignment2-GamePlay
git checkout VP/ml

echo.
echo ============================================
echo  Setup DONE! Run generation with:
echo  py -u generate_parallel.py --games 9000 --level 10 --think 3.0 --workers 60 --opening 4 --output Dataset\l10\data_l10_v3.npz
echo ============================================
pause
