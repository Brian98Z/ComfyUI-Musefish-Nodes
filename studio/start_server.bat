@echo off
rem ----------------------------------------------------------- Musefish Studio
rem Musefish 插件本地客户端：Studio 服务(17998) + headless ComfyUI(17996)
rem 重量级资产全部经 junction 映射自 H:\ComfyUI_Mie_2026_V8.0，本目录只存薄链接与产物。
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
set "PYTHON=%ROOT%runtime\python\python.exe"
if not exist "%PYTHON%" (
  echo [ERROR] 未找到内嵌 Python：%PYTHON%
  echo         先执行：python tools\link_models.py
  exit /b 3
)
if not exist "%ROOT%comfy\main.py" (
  echo [ERROR] 未找到 ComfyUI 映射目录：%ROOT%comfy
  echo         先执行：python tools\link_models.py
  exit /b 3
)
set "PYTHONIOENCODING=utf-8"
set "PATH=%ROOT%runtime\python;%PATH%"
"%PYTHON%" -u "%ROOT%server\app.py" %*
