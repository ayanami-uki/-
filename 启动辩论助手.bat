@echo off
title 辩论备赛助手
cd /d D:\Git\debate-helper
echo.
echo ============================================
echo    🎤 辩论备赛助手 启动中...
echo ============================================
echo.
start "" http://localhost:8501
D:\AI\python\python.exe -m streamlit run app.py --server.port 8501 --server.headless true
pause
