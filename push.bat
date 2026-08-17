@echo off
REM Simple one-click push script.
REM Usage: double-click this file, or run "push.bat Your commit message" from cmd.

setlocal
set MSG=%*
if "%MSG%"=="" set MSG=Update project files

git add .
git commit -m "%MSG%"
git push

echo.
echo Done. Press any key to close.
pause >nul
