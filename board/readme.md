# The Board Package
---
<sub>Glenn Abastillas | August 26, 2019 | The Board of Veterans' Appeals</sub>

This guide assumes you have access to the VEO databases and VSignals in addition to working within the VA network.

<a name="top"></a>
1. [Set Up](#setup)
      1. [Instructions](#what-to-do)
      2. [Folders](#folders)
      3. [Resources](#resources)
      4. [Dependencies](#dependencies)
2. [How to Use](#use)
      1. [Default Method](#cli)
      2. [Single Click](#alternative)
      3. [Command Line Tool](#clt)
3. [Script Settings](#settings)
      1. [Format](#setting_format)
      2. [Additions](#add)
      3. [Changes](#change)

<a name="setup"></a>
## 1. Set Up

You must have VA VEO database access permissions before you can run this script successfully. You must also be working within the VA network on site or through the VA VPN off site.

<a name="what-to-do"></a>
##### Instructions (What to Do)
Set up is a three step process.

 1. **Install** prerequisite [dependencies](#dependencies)
 2. **Clone** repo 
    * Git command: `git clone https://dev.azure.com/abastillas/bah/_git/board`
    * Regardless of where this repo is cloned, it will access resources [in your local `~/Documents` folder](#folders). These resources will be automatically created the first time you run the script.
 3. **Run** the `board` command ([learn how](#use))

<sub>&uarr; [Back to Main Menu](#top)</sub>

<a name="folders"></a>
##### Folders
This script will automatically create the folders it requires after running it for the first time. It uses the following folder structure nested in the User's Documents folder:
```
  ~/Documents/board/
                  ./data
                  ./docx
                  ./images
                  ./template
```

<a name="resources"></a>
###### Requisite Resources
Resources autopopulate after [running the script for the first time](#use). If the `~/Document/board` folder and subfolders do not appear, copy the repo file structure into `~/Document/board`.

<sub>&uarr; [Back to Main Menu](#top)</sub>

<a name="dependencies"></a>
##### Dependencies

This script requires these python packages to be installed:

   - pandas
   - numpy
   - python-docx
   - emoji
   - ftfy

###### Installation
Run either the `pip` or `conda` command in your command line.

`pip install --upgrade pandas numpy python-docx emoji ftfy`
<br/>`conda install -c conda-forge pandas numpy python-docx emoji ftfy`

<sub>&uarr; [Back to Main Menu](#top)</sub>

<a name="use"></a>
## How to Use

This distribution of the Board script is run using the *command line interface* (CLI). There are steps to take _before running the script_ and _after running the script_.

**Deprecated: 2019-09-23**
<span style="color: #cccccc;">
_Before Running the Script_, make sure the `~/Documents/board/data/Sattrackdist.csv` file is up to date with data from the Board dashboard from VSignals. You can download this file from VSignals by logging into the VSignals dashboard then completing the following steps:
  1. <span style="color: #cccccc;">Select Board User role</span>
  2. <span style="color: #cccccc;">Navigate to Score Distribution page from Data Analysis tab</span>
  3. <span style="color: #cccccc;">Download data as an Excel file by clicking on the ribbon at the top of the page</span>
  4. <span style="color: #cccccc;">Convert the `Sattrackdist.xls` file into a `Sattrackdist.csv` file</span>
  5. <span style="color: #cccccc;">Run script</span>

</span>

_After Running the Script_, these are some steps to complete:
  1. **Deprecated: 2019-09-23**<span style="color: #cccccc;">Update VSignals data
      - Replace A-11 Image placeholder
      - Update survey-specific Trust scores on last page</span>
  2. Curate Comments
      - Make sure there are at least 5 Compliments and 5 Concerns
      - Highlight <mark>relevant text</mark> as appropriate
  3. Fix any formatting issues:
      - Capitalization issues
      - Remove <mark>highlights</mark> from updated variables
      - On the first page:
          - Percentage differences with <span style="color:rgb(112,147,2)">+ should be green</span>
          - Percentage differences with <span style="color:rgb(188,79,7)">- should be red</span>
          - Gray out any <span style="color: #bbbbbb">0.00%</span> or <span style="color: #bbbbbb">NA</span> values
<a name="cli"></a>
##### Default Method

1. **Open** the command line or terminal
2. **Go to** to script folder
3. **Run** the script using `board` command

**Deprecated: 2019-09-23**
    - <span style="color: #cccccc;">_Example Use 1_: `board -e [end date] 
    -t [period] -s`</span>
        * <span style="color: #cccccc;">Example with all [flags](#flags). Query results will be stored in `~/Documents/board/data/query results`. All flags are optional.</span>
    - <span style="color: #cccccc;">_Example Use 2_: `board`
        * Example with no [flags](#flags). Default values will be used.</span>
    - <span style="color: #cccccc;">_Example Use 3_: `board -e 2018-12-01`
        * Example with the end date specified. Default value for `-t` or `--time-period` will be `week` for a weekly report.
    </span>

<a name='flags'></a>
###### Flags and Arguments
These flags are optional when using the `board` command.


Name | Flag | Argument Format | Default Value | Definition
--: | :-:  | :-: | :-: | :--
End Date | `-e` , `--end-date` | `YYYY-MM-DD` | Date run | Last date of the report. Data from this date is **excluded**
Time Period | `-t`,  `--time-period` | [`week`, `midmonth`, `month`] | `week` | Length of time to include for the comparison pages of the report
Store | `-s` | `None` | `False` (do not store) | Store query results locally in `~/Documents/board/data/query results`

<sub>&uarr; [Back to Main Menu](#top)</sub>

<a name="messages"></a>
###### Onscreen Messages
The CLI will display what stage of the process the script is in. Upon completion, the screen will display where the resulting Board report output is saved. This should be in either the `week`, `midmonth`, or `month` folder in `~/Documents/board/docx/`. 

Lastly, the screen will display the three data sets and their shapes `(# rows/respondents, # columns)` used to generate the report. The format is as follows:
```
   90 Day Period: (1234, 13) <-- (number of rows/respondents, number of columns)
  Current Period: (123, 13)
 Previous Period: (123, 13)
```

<sub>&uarr; [Back to Main Menu](#top)</sub>

<a name="alternative"></a>
##### Single Click Method
You can double click on the `board.bat` file and the `board` script will open a CLI and run using default values. That is, the end date will be the date of the script is executed and the time period will be `week`.

<sub>&uarr; [Back to Main Menu](#top)</sub>

<a name="clt"></a>
##### Command Line Tool Method
Update your system's `PATH` variable to include the folder that contains the `board.bat` file. Remember to use a semicolon with no trailing space when updating the `PATH` variable. 

  * `...;path/to/folder` <span style="color:green">&check;</span>
  * `...; path/to/folder` <span style="color:red">X</span> 

Run the `board` command from the command line window from any folder like in the [default method](#cli).

<sub>&uarr; [Back to Main Menu](#top)</sub>

<a name="tweaking"></a>
##### Manual Tweaking
Once the report is compiled, there are a few more steps to complete before the report is ready for final quality control.

These steps include:
    * Updating images
    * Curating comments
    * Finalizing formatting

##### Updating Images
This report makes use of 

<a name="settings"></a>
## Script Settings

Script settings are specified in `~/Documents/board/data/settings.txt`.

<a name="setting_format"></a>
##### Format
Each setting follows this JSON format:
```json
  {
    "SETTING": {
      "DATA": "--> data goes here <--",
      "DESC": "--> setting description goes here <--"
    },
    // etc. ...
  }
```
<sub>&uarr; [Back to Main Menu](#top)</sub>

###### Special Formats
There are settings that extend the basic scheme. These are: APG, QUERY, and URL.

**APG**
This setting defines tables to be used by queries in this package.
```json
  {
    "APG": {
      "DATA": {
          "TABLE": "APG.TABLE_NAME_HERE",
          ...
      },
      "DESC": "--> setting description goes here <--"
    },
    // etc. ...
  }
```

**QUERY**
This setting defines queries to be used by the board package.
```json
  {
    "QUERY": {
      "DATA": {
          "LABEL": "SELECT Query, Goes, Here FROM Settings",
          ...
      },
      "DESC": "--> setting description goes here <--"
    },
    // etc. ...
  }
```

**URL**
This setting defines VSignal relevant URLs and other URLs to be used by the package.
```json
  {
    "URL": {
      "DATA": {
          "LABEL": "https://url.goes.here",
          ...
      },
      "DESC": "--> setting description goes here <--"
    },
    // etc. ...
  }
```

<a name="add"></a>
##### Add a Setting
Use the setting format above to add a new setting directly into the `settings.txt` file.

<sub>&uarr; [Back to Main Menu](#top)</sub>

<a name="change"></a>
##### Change a Setting
Change either the `DATA` or `DESC` value according to the desired changes. For settings with filepaths, changing these filepaths in `DATA` will cause the script to look for the new filepath for resources. If the new filepath does not exist, it will create a new folder and populate the folders with the default resources from the cloned repo.

<sub>&uarr; [Back to Main Menu](#top)</sub>

---
# END

