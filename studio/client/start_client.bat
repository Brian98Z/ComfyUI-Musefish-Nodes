@echo off
rem ----------------------------------------------------------- Musefish Studio 客户端
rem 直接调用本地 electron.exe 加载 client/，双击即用，无需 npm。
rem 服务端由客户端自动拉起（server/app.py，17995）。
setlocal
chcp 65001 >nul
set "ROOT=%~dp0.."
set "ELECTRON=%ROOT%client\node_modules\electron\dist\electron.exe"
set "CLIENT=%ROOT%client"

if not exist "%ELECTRON%" (
  echo [ERROR] 未找到 Electron：%ELECTRON%
  echo         在 client 目录执行一次：npm install
  exit /b 3
)

start "" "%ELECTRON%" "%CLIENT%"
exit /b 0
