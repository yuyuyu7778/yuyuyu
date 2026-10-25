@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal enabledelayedexpansion

echo ============================================================
echo  Push D:\mmm  to  github.com/yuyuyu7778/yuyuyu
echo  branch: b-track  to  main   size: about 26 MB
echo ============================================================
echo.
echo   [1] Browser login  - Git Credential Manager popup
echo   [2] Paste token    - Personal Access Token
echo.
set /p MODE=Choose 1 or 2 then press Enter: 

if "%MODE%"=="2" goto TOKEN
goto BROWSER

:BROWSER
echo.
echo Opening credential window, please sign in as yuyuyu7778 ...
git push -u origin b-track:main
goto DONE

:TOKEN
echo.
set /p TK=Paste your token then press Enter: 
echo Pushing ...
git push https://%TK%@github.com/yuyuyu7778/yuyuyu.git b-track:main
set TK=
goto DONE

:DONE
echo.
echo ------------------------------------------------------------
echo Exit code: %errorlevel%
echo   0    = success, open https://github.com/yuyuyu7778/yuyuyu
echo   403  = wrong account, must sign in as yuyuyu7778
echo   128  = auth failed or network, check proxy 127.0.0.1:7897
echo ------------------------------------------------------------
pause
