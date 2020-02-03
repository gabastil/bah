@ECHO OFF
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
rem Batch File for the Board of Veterans' Appeals
rem Glenn Abastillas | October 3, 2019 | Analytics Report
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
rem
rem This batch file copies pyc files to the build/board directory
rem
rem - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

CALL compile
@ECHO OFF
ECHO Building package

rem Loop through all the compiled python files and copy binaries over to the
rem builds/board folder if they aren't already present. If overwrite is present
rem then overwrite existing copies of the binaries.

SET BOARD_FOLDER="\scripts\board"
SET BUILD_FOLDER="\scripts\builds\board"
SET USERS_FOLDER="%HOMEPATH%\Documents\board"

FOR /R ".\__pycache__" %%P IN (*.pyc) DO (
    ECHO ">>> Copying %%~P"
    IF "%1"=="--overwrite" (
        ECHO F|XCOPY /Q /Y "%%P" "%USERS_FOLDER%\%%~nxP"        
        rem ECHO F|XCOPY /Q /Y ""%%P"" "%HOMEPATH%\Documents\board\%%~nxP"        
    )
    ECHO F|XCOPY /Q /Y "%%P" "%BUILD_FOLDER%\%%~nxP"
        rem ECHO F|XCOPY /Q /Y ""%%P"" "..\builds\board\%%~nxP"    
)

rem Loop through folder labels and copy resource folders to /build/board
FOR /D %%I IN ("data", "docx", "images", "resources", "template") DO (
    ECHO ">>> Copying %%~I"

    rem If --overwrite flag is on, replace resources in /builds/board
    IF "%1" EQU "--overwrite" (

        rem Copy over only the template.docx and variables.xlsx files
        IF "%%~I" EQU "template" (
            FOR /F %%G IN ('DIR /A-D /B .\template') DO (
                ECHO F|XCOPY /Q /Y ".\%%~I"\%%~nxG "%BUILD_FOLDER%\%%~I\%%~nxG"
            )
        )

        rem Copy over only the files in ./data and none of the subfolders
        IF "%%~I" EQU "data" (
            FOR /F %%G IN ('DIR /A-D /B .\data') DO (
                ECHO F|XCOPY /Q /Y ".\%%~I"\%%~nxG "%BUILD_FOLDER%\%%~I\%%~nxG"
            )
        )

        rem Copy over only the files in ./data and none of the subfolders
        IF "%%~I" EQU "resources" (
            FOR /F %%G IN ('DIR /A-D /B .\resources') DO (
                ECHO F|XCOPY /Q /Y ".\%%~I"\%%~nxG "%BUILD_FOLDER%\%%~I\%%~nxG"
            )
        )

        rem Default operation for the other folders
        IF "%%~I" NEQ "data" (
            IF "%%~I" NEQ "template" (
                ECHO Y|DEL "%BUILD_FOLDER%\%%~I\"
                ECHO D|XCOPY /E /Q /Y ".\%%~I" "%BUILD_FOLDER%\%%~I\"
            )
        )

    ) ELSE (
        rem Default operation for folders if --overwrite is not on
        IF NOT EXIST "%BUILD_FOLDER%\%%~I\" (    
            ECHO Y|DEL "%BUILD_FOLDER%\%%~I\"
            ECHO D|XCOPY /Q /Y ".\%%~I" "%BUILD_FOLDER%\%%~I\"
        )
    )
)

rem Copy over the latest version of the board.bat script
ECHO F|XCOPY /S /Q /I /Y /F "%BOARD_FOLDER%\board.bat" "%BUILD_FOLDER%\board.bat"
ECHO Build complete
ECHO ON