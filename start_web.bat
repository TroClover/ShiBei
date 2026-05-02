@echo off
chcp 65001 >nul
set NO_PROXY=localhost,127.0.0.1
call E:\Anaconda3\Scripts\activate.bat shibei
cd /d E:\Project\CC\ShiBei
echo ============================================
echo   ShiBei 文档知识库系统
echo   Provider: %PROVIDER%
echo ============================================
echo.
start http://localhost:8501
streamlit run web_app.py
pause
