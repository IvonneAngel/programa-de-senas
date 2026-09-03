@echo off
setlocal
cd /d "%~dp0"

chcp 65001 >nul
title FEMECI - Traductor de señas
mode con: cols=104 lines=42 >nul 2>nul

set "MENU=%~dp0panel del traductor.ps1"
set "POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if not exist "%MENU%" (
  echo.
  echo [traductor] No se encontro panel del traductor.ps1 junto a este lanzador.
  echo Ruta esperada: "%MENU%"
  echo.
  pause
  exit /b 1
)

"%POWERSHELL%" -NoProfile -ExecutionPolicy Bypass -File "%MENU%" -Web %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo [traductor] El panel termino con error. Revisa la salida anterior.
  echo.
  pause
)

exit /b %EXIT_CODE%
