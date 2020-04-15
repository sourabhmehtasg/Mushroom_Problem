@echo off

echo working directory is: 
echo %cd%

python -m venv venv

call venv/Scripts/activate.bat

pip install -r requirements.txt

set "x=%cd%"
echo "%x%"
cd ..
set "y=%cd%"
echo "%y%"

python %x%/app.py %y%
cd App
call deactivate

