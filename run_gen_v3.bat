@echo off
echo Starting data generation...
echo Level 10, Think 3.0s, 60 workers, 9000 games
echo Output: Dataset\l10\data_l10_v3.npz
echo.
py -u generate_parallel.py --games 9000 --level 10 --think 3.0 --workers 60 --opening 4 --output Dataset\l10\data_l10_v3.npz
echo.
echo Done!
pause
