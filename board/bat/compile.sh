# compile.sh
# Glenn Abastillas
# September 28, 2019
# This script is a tester script for compiling the specified python files
# within this folder.
#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

echo Compiling files
python -m py_compile calculate.py canvas.py config.py database_access.py writer.py
python -m renamer
echo Compilation complete
