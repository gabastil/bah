@echo off
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
rem Batch File for the Board of Veterans' Appeals
rem Glenn Abastillas | August 27, 2019 | Analytics Report
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
rem
rem This batch script enables command line usage of the Board script.
rem This script assumes that `python` is a callable command in your terminal.
rem
rem How to Use:
rem
rem        Example #1
rem
rem     >> board -e 2019-08-27 -t week -s
rem     >> # This will create a weekly board report that ends
rem     >> # on August 27, 2019 and will save the data pulled
rem     >> # from the database as a separate file in the data
rem     >> # folder.
rem     >>
rem     >> # This ability has been *disabled* for current
rem     >> # distributions of this script.
rem
rem        Example #2
rem
rem     >> board -e 2019-08-27 -t week
rem     >> # This will do the same as above except the data
rem     >> # pulled from the database will not be saved as a
rem     >> # separate file in the local data folder.
rem
rem        Example #3
rem
rem     >> board -e 2019-08-27
rem     >> # This will do the same as Example #2 (above). The
rem     >> # default value for -t is already `week`. The user
rem     >> # must specify if -t is to be `midmonth` or `month`
rem
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
SET USERS_FOLDER="%HOMEPATH%\Documents\board"
CD %USERS_FOLDER%
rem PUSHD %USERS_FOLDER%
python Report.pyc %1 %2 %3 %4 %5 %6 %7 %8 %9
rem POPD
PAUSE
ECHO on