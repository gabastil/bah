@echo off
rem pull.bat
rem Sept. 26, 2019
rem Glenn Abastillas
rem Description: Batch script to automate adding and committing current changes
rem prior to pulling from the set upstream repo.
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

git add * && git commit -m %1 && git pull
