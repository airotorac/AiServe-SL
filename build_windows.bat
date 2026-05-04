@echo off
echo === AiServe SL Windows Build ===
echo Installing dependencies...
pip install -r requirements.txt
echo Building EXE...
pyinstaller --onefile --windowed --name "AiServe SL" --clean aiserve.py
echo.
echo Done! EXE is in: dist\AiServe SL.exe
pause
