@echo off
:: ============================================================
:: server_run.bat — Chay parallel benchmark tren Windows server
:: Usage: chinh sua cac bien WORKERS, GAMES, TIME ben duoi
:: ============================================================

set MODEL=models\best_mlp_model (7).pt
set WORKERS=60
set GAMES=200
set TIME=3.0
set LEVELS=0,1,3,5,7,9

echo ========================================
echo  Parallel Benchmark - ML vs Minimax
echo  Model  : %MODEL%
echo  Workers: %WORKERS%
echo  Games  : %GAMES% per matchup
echo  Time   : %TIME%s per move
echo  Levels : %LEVELS%
echo ========================================
echo.

py -u benchmark\parallel_benchmark.py ^
    --model "%MODEL%" ^
    --workers %WORKERS% ^
    --games %GAMES% ^
    --time %TIME% ^
    --levels %LEVELS%

echo.
echo ========================================
echo  DONE! Ket qua luu o benchmark\benchmark_parallel.png
echo ========================================
pause
