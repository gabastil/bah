@echo off
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
rem Batch File for the Board of Veterans' Appeals
rem Glenn Abastillas | September 12, 2019 | Analytics Report
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
rem
rem This batch script compiles Python scripts for distribution.
rem
rem How to Use:
rem
rem        Example #1
rem
rem     >> compile
rem     >>
rem     >> Compilation complete
rem
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
pushd .\__pycache__
del "*.pyc"
popd

python -m py_compile __init__.py calculate.py canvas.py configuration.py database.py Report.py writer.py

pushd .\__pycache__
rename "??????????????????.cpython-37.pyc" "??????????????????.pyc"
popd

if %ERRORLEVEL%==0 (echo Compilation complete) else (echo Compilation error)
echo on