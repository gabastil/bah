#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:57:59 2019
@author: abastillasgl

CANVAS SCRIPT
-------------------------------------------------------------------------------
Glenn Abastillas | September 16, 2019 | canvas.py
-------------------------------------------------------------------------------

Description:

These methods create graphs that correspond to the graphics pulled from 
VSignals data.

"""

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors
from configuration import Configuration
#import pandas as pd
import numpy as np
#from datetime import datetime
import os

config = Configuration()
#plot_path = Path("./images/a11_drivers.png")        

def hex2hsv(value):
    """ Convert a hex string to an hsv numpy array """
    return colors.rgb_to_hsv(colors.hex2color(value))

def hsv2hex(value):
    """ Convert an hsv numpy array to a hex string """
    return colors.rgb2hex(colors.hsv_to_rgb(value))

def saturate(value, amount):
    """ Change the saturation (+/-) of an input hex color. """
    hue, saturation, value = hex2hsv(value)
    if amount > 0:
        saturation = min(1, saturation + amount)
    else:
        saturation = max(0, saturation + amount)
    return hsv2hex([hue, saturation, value])

def saveplot_A11(series, 
                 output=config.paths.a11_image, 
                 figsize=(15, 4.5), 
                 offset=3, 
                 show=False, 
                 save=True):
    """
    Create and save A-11 CX Domain barplot resembling the one on VSignals
    
    Parameters
    ----------
        series (Series): A-11 CX Domains (index) and Agreement scores (values)
        output (Path): path to output image
        figsize (tuple): size of the plot
        offset (float): number of percentage points to offset each label
        show (bool): Show the plot? Default is "No" (False).
        save (bool): Save the plot? Default is "Yes" (True).

    Example
    -------
        >>> saveplot_A11(data['a11'])
        >>> # The plot will be saved to the config.paths.a11_image path
        >>> # The default output location is:
        >>> #     ~/Document/board/images/a11_figure.png
        >>>
    """
    HEX, ORDER = config.format.a11_hex, config.columns.a11_order
    sns.set(font_scale=1.2, style='whitegrid')
    sns.set_palette(HEX)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.barplot(series.index, series.Score, order=ORDER, ax=ax)

    for i, driver in enumerate(ORDER):
        x, y = i, series.loc[driver].Score
        label = round(y, 1)
        ax.annotate(label, (x, y + offset), ha='center')
    ax.set_xlabel("A-11 CX Domain")
    ax.set_ylabel("Percentage Agreement (4 - 5)", fontsize='large')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    yticklabels = [f"{n-10}%" for n in range(0, 111, 10)]
    ax.set_yticklabels(yticklabels, fontsize='large')

    ax.set_xlabel("A-11 CX Domain")
    plt.xticks([])
    
    # This code adds a legend to the bottom of the figure
    handles = []
    for i, (color, label) in enumerate(zip(HEX, ORDER)):
        current_label = Patch(color=color, label=label.replace("_", "/"))
        handles.append(current_label)
        
    plt.legend(handles=handles, 
               bbox_to_anchor=(.5,-.2), 
               frameon=False,
               loc='center', 
               ncol=7)

    if save:
        if output.exists():
            os.remove(output)
        plt.savefig(output, bbox_inches='tight')
    if show:
        plt.show()

def saveplot_TOT(df, 
                 output=config.paths.tot_image, 
                 figsize=(15, 10), 
                 offset=None, 
                 show=False, 
                 save=True):
    """
    Create and save a Trust-over-Time time series by Survey Type
    
    Parameters
    ----------
        df (DataFrame): SurveyType data with ResponseDateTime (e.g, tot)
        output (Path): path to output image
        figsize (tuple): size of the plot
        show (bool): Show the plot? Default is "No" (False).
        save (bool): Save the plot? Default is "Yes" (True).
    
    Example
    -------
        >>> saveplot_TOT(data['tot'])
        >>> # The plot will be saved to the config.paths.tot_image path
        >>> # The default output location is ~/Document/board/images/a11_figure.png
        >>>
    """

    # General plot settings and figure definition
    sns.set(font_scale=1, style='whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the time series line graph
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 11
    sns.lineplot(data=df, ax=ax, 
                 x='Date', 
                 y='Trust', 
                 hue='SurveyType', 
                 legend=False,
                 markers=True,
                 dashes=False,
                 style=2,
                 units='SurveyType',
                 estimator=None,
#                 hue_order=config.columns.tot_order,
                 palette=config.format.tot_hex)
 
    annotation = dict(ha='center', fontsize='large')
    
    # Annotate chart with labels and arrows pointing to the first date
    for surveyType in df['SurveyType'].unique():
        condition = (df['SurveyType'] == surveyType)
        current_df = df[condition][['Date', 'Trust']]
        
        x, y = current_df.iloc[0, :2]

        index = config.columns.tot_order.index(surveyType)
        color = config.format.tot_hex[index]

        arrowprops = dict(arrowstyle='-', lw=1, ls='-', color=color)
        
        annotation['arrowprops'] = arrowprops
        annotation['xytext'] = (x, y - 7 if y < 45 else y + 7)
        annotation['xy'] = (x, y)
        annotation['s'] = surveyType

        ax.annotate(**annotation)
    
    # Time series X axis format
    ax.set_xlabel("")
    ax.set_xticks([])

    # Time series Y axis format
    ax.set_ylabel("Trust Agreement (4-5)", fontsize='x-large')
    ax.set_ylim(-1, 100)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    yticklabels = [f"{n-10}%" for n in range(0, 111, 10)]
    ax.set_yticklabels(yticklabels, fontsize='large')
    
    # Create Data Table for beneath the chart
    pivot = df.pivot(index='SurveyType', 
                     columns='Date', 
                     values='Trust')
    pivot.columns = df.label.unique()
    pivot = pivot.loc[config.columns.tot_order]
    
    # Plot the Data Table beneath the time series
    table = plt.table(cellText=pivot.values,
                      rowLabels=pivot.index,
                      colLabels=pivot.columns,
                      loc='bottom',
                      rowLoc='right',
                      colLoc='center',
#                      rowColours=[saturate(hex_, .1) for hex_ in config.format.tot_hex],
                      bbox=(0,-.2,1,.2))

    # Loop through the cells in the table to format them
    table_properties = table.properties()
    table_cells = table_properties['child_artists']
    
    #              Fill Colors for Data Table
    #              Lowest     Low-Mid    Middle     High-Mid   Highest  
    #              0.0%       1-25%      26-50%     51-75%     76-100%
    fill_colors = config.format.tot_table

    for cell in table_cells:
        text = cell._text.get_text().strip()
        cell.set_edgecolor("#cccccc")
        cell.set_text_props(ha='center')
        cell.set_text_props(fontsize='x-large')

        # Format Survey Type labels
        if text in config.columns.tot_order:
            index = config.columns.tot_order.index(text)
            color = saturate(config.format.tot_hex[index], 0.25)
#            cell._text.set_color(color)
            cell.set_text_props(weight='bold')

        # Format NOD and Hearing labels to have white font
#        if text in config.columns.tot_order[::3]:
#            cell._text.set_color("#ffffff")

        # Remove text from cells with `nan` values
        if text == 'nan':
            cell.set_facecolor('#dddddd')
            cell._text.set_text("")

        # Fill cell based on Trust value of cell
        if text.replace('.', '').isnumeric():            
            number = float(text)

            if number == 0.0:
                cell.set_facecolor(fill_colors[0])
            else:
                c_i = np.digitize([number], range(0, 101, 25))[0]
                cell.set_facecolor(fill_colors[c_i])
            
            cell._text.set_text(f"{number}%")
    
    if save:
        if output.exists():
            os.remove(output)
        plt.savefig(output, bbox_inches='tight')
    if show:
        plt.show()
