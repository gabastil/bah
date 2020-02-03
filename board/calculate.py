#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:33:28 2018
@author: glenabastillas

CALCULATE SCRIPT
-------------------------------------------------------------------------------
Glenn Abastillas | November 14, 2018 | calculate.py
-------------------------------------------------------------------------------

Description:

This script contains methods to perform transformations to string and numeric
data. These data typically come as a result of data pulls from the sister
database_access.py script.

Non-standard library dependencies:
    numpy
    pandas
    ftfy
    config (custom)

Update:
    9/13/2019: Added calculate_A11() method to get Trust values by A-11 drivers
    9/26/2019: Added functions to derive best and worst questions from VEODB
    9/30/2019: Removed the sattrackdist functions now that we are using VEODB

"""
from datetime import datetime, timedelta, date
from configuration import Configuration
import calendar
import numpy as np
import pandas as pd
import canvas
import ftfy
import re

config = Configuration()

"""
Function Group: load_*

Functions to load resources required by other functions in this script
"""


def load_A11(path=config.paths.a11, *args, **kwargs):
    """
    Load the A11 CX Domain drivers from ./data

    Parameters
    ----------
        path (str): Path to A11 CX Domain file
        args (list, tuple): positional arguments for pd.read_csv
        kwargs (dict): keyword arguments for pd.read_csv

    Returns
    -------
        DataFrame of A11 CX Domains
    """
    return pd.read_csv(path, *args, **kwargs)


def load_representatives(path=config.paths.reps, is_pickle=True):
    """
    Load representative data from ./data. Default path is config.paths.reps

    Parameters
    ----------
        path (Path, str): path to pickle or csv with representatives data
        is_pickle (bool): Is the target path file a pickled file or csv?

    Returns
    -------
        Mapping (dict) of representatives to their acronyms or abbreviations
    """
    if is_pickle:
        representatives = pd.read_pickle(path)
    else:
        representatives = pd.read_csv(path)
    keys = [_.lower() for _ in representatives.keys()]
    return dict(zip(keys, representatives.values()))


def load_periods_of_service(path=config.paths.wars, is_pickle=True):
    """
    Load representative data from ./data. Default path is config.paths.wars

    Parameters
    ----------
        path (Path, str): path to pickle or csv with Period of Service data
        is_pickle (bool): Is the target path file a pickled file or csv?

    Returns
    -------
        Mapping (dict) of periods of service to their acronyms or abbreviations
    """
    if is_pickle:
        return pd.read_pickle(path)
    return pd.read_csv(path)


def load_topics(path=config.paths.topics, is_pickle=False):
    """
    Load topic data from ./data. Default path is config.paths.topics

    Parameters
    ----------
        path (Path, str): path to pickle or csv with topic data
        is_pickle (bool): Is the target path file a pickled file or csv?

    Returns
    -------
        Listing of topics (DataFrame), their descriptions, regular expressions,
        and feature names.
    """
    if is_pickle:
        return pd.read_pickle(path)
    return pd.read_csv(path)

"""
Function Group: calculate_*

History: Functions to Create Tables for The Board. These functions come
from the Jupyter Notebook "Report Builder" on AWS.
"""

def calculate_A_B(df, column, scores=config.lists.agree_scores, **kwargs):
    """
    Get the numerator and denominator for a calculation

    Parameters
    ----------
        df (DataFrame): data with response data from a 5-point Likert scale
        column (str): name of column with responses
        scores (list): Valid scores
        kwargs (dict): keyword arguments for filtering data set.

            Numerator Keyword Arguments
            ---------------------------

            `isin` (bool): requires values to be elements of a list
                `---> `scores` (list, set): elements to include

            `startswith` (bool): checks if string starts with a sequence
                `---> `string` (str): sequence of characters

            `endswith`: checks if string ends with a specified sequence
                `---> `string` (str): sequence of characters

            `agg`: run aggregate function over data
                `---> `func` (function): function to run over data

            Denominator Keyword Arguments
            -----------------------------

            `notnull` (bool): include any value that is not null (pd.notnull)
            `count` (bool): include any value counted in groupby().count

    Returns
    -------
        Numerator (int) and denominator (int)
    """

    if kwargs.get('startswith', None):
        string = kwargs.get('string', None)
        assert string
        A = df[column].str.startswith(string)

    elif kwargs.get('endswith', None):
        string = kwargs.get('string', None)
        assert string
        A = df[column].str.endswith(string)

    elif kwargs.get('agg', None):
        func = kwargs.get('func', None)
        assert func
        A = df.agg(func)

        if kwargs.get('agg', None) == 'sum':
            A = A.sum()
        elif kwargs.get('agg', None) == 'mean':
            A = A.mean()
        elif kwargs.get('agg', None) == 'count':
            A = A.count()

    else:
        A = df[column].isin(scores)

    if kwargs.get('count', None):
        B = df.count()
        return A, B
    else:
        B = df[column].notnull()

    A, B = df[A], df[B]
    return number_of_respondents(A), number_of_respondents(B)


def calculate_scores(df, column=None, scores=config.lists.agree_scores):
    """
    Calculate the proportion of responses that match the specified scores

    Parameters
    ----------
        df (DataFrame): data with response data from a 5-point Likert scale
        column (str): name of column with responses
        scores (list): Valid scores

    Returns
    -------
        Decimal score showing proportion of variable matching specified scores
    """
    A, B = calculate_A_B(df, column=column, scores=scores)
    if (B is None) or (B == 0):
        return None
    return A / B


def calculate_agrees(df, column, agrees_scores=config.lists.agree_scores):
    """ Get proportion of responses that are either 4 or 5 """
    return calculate_scores(df, column, agrees_scores)


def calculate_disagrees(df, column, disagree_scores=config.lists.disagree_scores):
    """ Get proportion of responses that are either 1 or 2 """
    return calculate_scores(df, column, disagree_scores)


def calculate_trust_by(df, column):
    """
    Calculate the average Trust score for a DataFrame grouped by a column.

    Parameters
    ----------
        df (DataFrame): Query result data with Trust scores
        column (str): Column name to group by

    Returns
    -------
        DataFrame with average Trust for each value.
    """
    grouped = df.groupby([column])
    return grouped.agg(calculate_trust)['Trust']


def calculate_trust(df, column='Trust', agree_scores=config.lists.agree_scores):
    """
    Calculate the overall Trust score for a DataFrame.

    Parameters
    ----------
        df (DataFrame): Query result data with Trust scores
        column (str): Name of column that contains Likert scores
        agree_scores (list): Valid "Strongly Agree" or "Agree" scores

    Returns
    -------
        Trust score percentage
    """
    trust = calculate_agrees(df, column, agree_scores)
    if trust is None:
        return None
    return 100. * trust


def calculate_trust_by_question(df, scores=config.lists.disagree_scores):
    """
    Get Trust by BVA Survey Question for A-11 CX Domain driver analysis.
    Requires auxiliary columns `SurveyPersonID` and `Trust`.

    **NOTICE**
    By default, this function calculates the percentage of 1 and 2 responses,
    i.e., "Strongly Disagree" and "Disagree".

    Parameters
    ----------
        df (DataFrame): Query result data with Survey Question data
        scores (list): Valid scores

    Returns
    -------
        Table (DataFrame) of A-11 CX Drivers and their associated Trust scores
    """
    drivers = load_A11().drop('Trust', axis=1, errors='ignore')
    drivers = drivers[config.columns.bva_survey]

    current = df[drivers.QuestionReference].agg(pd.to_numeric)
    disagrees = current.agg(lambda x: x.isin(scores))
    A, B = disagrees.sum(), current.count()

    scores = (A / B * 100).round(1).reset_index(name='Score')
    merged = drivers.merge(scores, left_on='QuestionReference', right_on='index')
    return merged.drop('index', axis=1)


def calculate_disagrees_by_question(df):
    return calculate_trust_by_question(df, scores=config.lists.disagree_scores)


def calculate_agrees_by_question(df):
    return calculate_trust_by_question(df, scores=config.lists.agree_scores)


def calculate_tot(df):
    """
    Calculate Trust by category for raw `tot` query results.

    Parameters
    ----------
        df (DataFrame): Query result data with Trust scores (e.g., tot)

    Returns
    -------
        Trust score percentage by group
    """
    grouped = df.groupby(config.columns.tot_groupby)
    tot = grouped.Agree.sum().reset_index()
    tot = tot.assign(Trust=round(100. * tot.Agree / tot.Total, 1),
                     Date=(tot.Year.astype('str') + '-' +
                           tot.Month.astype('str').str.zfill(2) + '-01'))
    make_label = lambda x: f"{date_format(x, asdatetime = True):%b '%y}"
    tot = tot.assign(label=tot.Date.apply(make_label))
    tot = tot[['SurveyType', 'Date', 'label', 'Trust']]

    # Check for duplicate aggregate Trust scores
    DUPLICATE_GROUPBY = ['SurveyType', 'Date']
    duplicate_check = tot.groupby(DUPLICATE_GROUPBY).count() > 1
    duplicate_found = duplicate_check.max()[0]

    # Resolve duplicate values
    if duplicate_found:
        duplicate_values = duplicate_check[duplicate_check].dropna()
        duplicate_values = duplicate_values.reset_index()

        duplicate_indices = []
        new_values = []

        # Loop through the duplicate list and remove them from the original df
        # Create new Trust scores by average duplicate ones
        for survey_, date_ in duplicate_values[DUPLICATE_GROUPBY].values:
            survey_match = tot.SurveyType == survey_
            date_match = tot.Date == date_
            current_indices = tot[survey_match & date_match].index.tolist()
            duplicate_indices.append(current_indices)

            subset = tot.loc[current_indices]
            trust_ = round(subset.Trust.mean(), 1)
            label_ = subset.iloc[0].label

            new_values.append([survey_, date_, label_, trust_])
            tot.drop(current_indices, errors='ignore', inplace=True)

        tot_addendum = pd.DataFrame(new_values, columns=tot.columns.tolist())
        tot = pd.concat([tot, tot_addendum])
        tot = tot.sort_values(DUPLICATE_GROUPBY).reset_index(drop=True)

    # CONSIDER DOING THE SUBSETTING IN THE CANVAS MODULE
    # Remove current month and subset TOT dataset to last 12 months
    tot['Date'] = tot['Date'].astype('<M8[ns]')
    y_, m_, __ = date_format(config.today, astuple=True)

    # Define time booleans
    THIS_MONTH = date_format((y_, m_, 1), asdatetime=True)
    A_YEAR_AGO = date_delta(THIS_MONTH, months=12, asdatetime=True)[-1]
    START = tot.Date >= pd.Timestamp(A_YEAR_AGO)
    END = tot.Date < pd.Timestamp(THIS_MONTH)
    LAST_12_MONTHS = START & END

    tot = tot[LAST_12_MONTHS]
    tot['Date'] = tot['Date'].apply(date_format)

    return tot


def isolate_a11_drivers(df, usecols=config.lists.a11_columns):
    """
    Subset the dataset to get A11 CX Domain questions only

    Parameters
    ----------
        df (DataFrame): Query result data with Survey Question data
        path (str): Path to A11 CX Domain questions
        usecols (list): List of columns to use in A11 CX Domain file

    Returns
    -------
        DataFrame with A11 CX Domain questions and a subset dataset
    """
    drivers = load_A11(usecols=usecols)
    subset = df[drivers.QuestionReference].agg(pd.to_numeric)
    return drivers, subset


def calculate_a11q(df, agree_scores=config.lists.agree_scores):
    """
    Get A11CXDomain scores using Question-Level top-box aggregation.

    Parameters
    ----------
        df (DataFrame): Query result data with Survey Question data
        agree_scores (list): Valid "Strongly Agree" or "Agree" scores

    Returns
    -------
        Pandas Series with A11CXDomain as the index and the percent agree score
        as the values.
    """

    def aggregate(column, agree_scores=agree_scores):
        """ Helper function to aggregate percent agreement for a domain """
        A = column.isin(agree_scores).sum()
        B = column.count()
        if B:
            return A / B * 100
        return pd.np.nan

    drivers, current = isolate_a11_drivers(df)
    current = current.agg(aggregate).reset_index(name='Score')
    merged = current.merge(drivers,
                           left_on='index',
                           right_on='QuestionReference')
    grouped = merged.groupby('A11CXDomain').mean()

    return grouped.round(1)


def calculate_a11r(df, agree_scores=config.lists.agree_scores, threshold=3.5):
    """
    Get A11CXDomain scores using Record-Level top-box aggregation.

    Parameters
    ----------
        df (DataFrame): Query result data with Survey Question data
        agree_scores (list): Valid "Strongly Agree" or "Agree" scores
        threshold (float): Minimum average value required to be counted

    Returns
    -------
        Pandas Series with A11CXDomain as the index and the percent agree score
        as the values.
    """
    drivers, current = isolate_a11_drivers(df)
    domains = drivers.A11CXDomain.unique()
    scores = []

    for domain in domains:
        questions = drivers[drivers.A11CXDomain == domain].QuestionReference

        # Calculate mean Likert score and count respondents with an average
        # greater or equal to the threshold
        sub = current[questions].T.mean()
        A, B = (sub >= threshold).sum(), sub.count()
        scores.append([domain, A / B])

    output = pd.DataFrame(scores, columns=['A11CXDomain', 'Score'])
    grouped = output.groupby('A11CXDomain').mean() * 100
    return grouped.round(1)


def calculate_a11(df, method='r', agree_scores=config.lists.agree_scores):
    """
    Get agree scores by BVA Survey Question for A-11 CX Domain driver analysis.

    Parameters
    ----------
        df (DataFrame): Query result data with Survey Question data
        method (str): `Record-level` or `question-level` top-box aggregate
        agree_scores (list): Valid "Strongly Agree" or "Agree" scores

    Returns
    -------
        Table (DataFrame) of A-11 CX Drivers and their associated agree scores

    Notes
    -----
        Record-level top-box shows to be closer to VSignals
        Question-level top-box shows to have lower values than VSignals
        
        Default calculation method is Record-level or `r`

    Log
    ---
        2019-12-16: Currently, method `r` most closely resembles VSignals.
        2019-11-14: Previously calculate_a11r reflected what is on VSignals
                    Currently, calculate_a11q reflects what is on VSignals
                    Changed method from 'r' to 'q'.
    """
    if method.lower().startswith('q'):
        return calculate_a11q(df, agree_scores=agree_scores)
    elif method.lower().startswith('r'):
        return calculate_a11r(df, agree_scores=agree_scores)
    raise ValueError(f"`method` must be in {{r, q}}. Currently `{method}`.")

"""
Function Group: assign_*

Description: These function are used to normalize data in a DataFrame.
"""


def assign_age_group(age, breaks=config.lists.age_breaks):
    """
    Determine what Age Group an age belongs to

    Parameters
    ----------
        age (int): Age to assign to a group
        breaks (list): list of ages to define age groups.

    Returns
    -------
        Age Group label
    """
    if age < breaks[1]:
        return f"{breaks[1] - 1} and below"
    elif age >= breaks[-2]:
        return f"{breaks[-2]} and above"
    else:
        for i, age_break in enumerate(breaks):
            if age < age_break:
                return f"{breaks[i - 1]} - {age_break - 1}"


def assign_age_groups(df, column="Age", name="Age_Group"):
    """
    Apply the `assign_age_group` function to a column in a DataFrame.

    Parameters
    ----------
        df (DataFrame): Data with ages to be assigned to age groups
        column (str): Column with age data
        name (str): New age group column name

    Returns
    -------
        DataFrame with new Age_Group column
    """
    return df.assign(**{name: df.Age.apply(assign_age_group)})


def consolidate_age_groups(df, column='Age_Group', limit=10):
    """
    Combine age group counts if number of respondents in a group does not meet
    the threshold limit. Default column of analysis is the Age_Group column
    after build_table(*args).

    Parameters
    ----------
        df (DataFrame): Data from build_table(*args, **kwargs)
        column (str): Column with age groups to be consolidated
        limit (int): minimum number of observations before consolidation done

    Returns
    -------
        Table with consolidated categories and updated values
    """
    df = df.sort_values(column, ascending=False).reset_index(drop=True)

    # Determine which indices need to be consolidated---are less than the limit
    indices = df[df.iloc[:, 1] < limit].index

    # If the indices are empty, return the original df
    if indices.size == 0:
        return df

    # If there is only one index, check if it is the first or last.
    # If first, set the second index to be the second row
    # If last , set the first index to be the second to last row
    elif indices.size == 1:
        if indices[0] == df.shape[0] - 1:
            combination_pairs = [[indices[0] - 1, indices[0]]]
        else:
            combination_pairs = [[indices[0], indices[0] + 1]]

    # If there are two indices, check if they are the first and last.
    # Insert the second and second-to-last indices for each index.
    elif indices.tolist() == [0, df.shape[0] - 1]:
        combination_pairs = [[0, 1], [df.shape[0] - 2, df.shape[0] - 1]]

    # Check if the index is the last index
    elif indices.tolist() == [df.shape[0] - 1]:
        combination_pairs = [[df.shape[0] - 2, df.shape[0] - 1]]

    # Normal index pairing algorithm: Check if the subsequent row
    # is the next integer in a sequence.
    # If yes, continue the loop;
    # If no , set the index as the second index in the pair and restart.
    else:
        i = 0
        pair, combination_pairs = [], []

        while i < indices.size - 1:
            current = indices[i]
            next_   = indices[i + 1]

            if len(pair) == 0:
                pair.append(current)

            if current + 1 != next_:
                pair.append(current)
                combination_pairs.append(pair)
                pair = []

            i += 1

        pair.append(indices[i])
        combination_pairs.append(pair)

    new_df = pd.DataFrame(columns=df.columns)

    for pair in combination_pairs:
        try:
            subset = df.loc[range(pair[0], pair[1] + 1)]
        except(IndexError):
            subset = df.loc[range(pair[0], pair[1])]

        # Calculate new counts and Trust scores
        new_count = subset.iloc[:, 1].sum()
        weights = subset.iloc[:, 1]/new_count
        new_trust = round((weights * subset.Trust).sum(), 1)

        # Create new labels for the consolidated age groups
        label_A = str(subset.iloc[0, 0])
        label_B = str(subset.iloc[-1, 0])

        new_label = f"{label_B[:2]}{label_A[2:]}"
        if label_B[-2:].isalpha() and label_B[:2] == '80':
            new_label = f"{label_A[:2]}{label_B[2:]}"
        elif label_B[-2:].isalpha():
            new_label = f"{label_A[-2:]}{label_B[2:]}"

        new_row = [[new_label, new_count, new_trust]]
        new_df = pd.concat([new_df, pd.DataFrame(new_row, columns=df.columns)])

    indices = [__ for pair in combination_pairs for __ in pair]

    df = df.drop(indices, errors='raise')
    new_df = pd.concat([new_df, df])
    return new_df.sort_values(column, ascending=False).reset_index(drop=True)


def filter_out_non_responses(df, column, values=config.lists.non_response):
    """
    Remove any responses that qualify as non-responses in a data frame.

    Parameters
    ----------
        df (DataFrame): Data with non-responses
        column (str): Column to filter by.
        values (list): String stems representing non-responses.

    Returns
    -------
        DataFrame without non-responses

    Notes
    -----
        A String stem are the characters that define a non-response.

        For example, 'Declined to' will make 'Declined to respond' and
        'Declined to answer' return as True
    """
    def filter_(text, string_stems=values):
        """ Filtering function for a single non-response value """
        if text:
            for stem in string_stems:
                if str(text).lower().startswith(stem):
                    return False
            return True
        return False

    valid_responses = df[column].apply(filter_)
    return df[valid_responses]


def build_table(df, column, limit=None, sortby='Trust', rm_na=True):
    """
    Calculate Trust score percentages grouped by a specified column

    Parameters
    ----------
        df (DataFrame): Data with categorical variables and Trust scores
        column (str): Column with categorical variables
        limit (int): Minimum number of respondents
        sortby (str): Name of the final column to sort by
        rm_na (bool): Remove blanks and NaNs in output table?

    Returns
    -------
        DataFrame with values, counts, and Trust scores.
    """
    # Unfurl lists
    unfurl = lambda x: x[0] if isinstance(x, list) else x
    unfurled = df[column].apply(unfurl)

    df = df.assign(**{column: unfurled})

    if rm_na:
        df = filter_out_non_responses(df, column)
    
    # 2019-12-20 Raise error to address empty tables.
    if df.shape[0] == 0:
        error_message = config.script.calculate.build_table.error.format(column)
        raise ValueError(error_message)

    # Create a DataFrame from value counts and rename a column
    counts = pd.DataFrame(df[column].value_counts()).reset_index()
    counts.columns = [column, 'Number of Respondents']

    # Calculate the Trust score for values in the same column
    trust_table = calculate_trust_by(df, column).apply(lambda x: round(x, 1))

    # Merge counts and Trust scores and sort by Trust score descending
    merged = counts.merge(trust_table, on=column)
    merged.sort_values(by=[sortby], ascending=False, inplace=True)

    if column.lower() in {'age_group', 'age group'}:

        max_respondents = merged['Number of Respondents'].max()
        min_respondents = merged['Number of Respondents'].min()

        limit = 10 if limit is None else limit
        consolidated = consolidate_age_groups(merged, column, limit)

        if max_respondents < 10:
            return consolidated

        # Keep consolidating until the minimum number of respondents >= 10
        max_iter = 100
        while min_respondents < 10:
            consolidated = consolidate_age_groups(consolidated, column, limit)
            min_respondents = consolidated['Number of Respondents'].min()

            max_iter -= 1
            if max_iter < 0:
                break

        return consolidated

    elif limit:
        merged = merged.loc[merged['Number of Respondents'] > limit, :].copy()

    merged.reset_index(drop=True, inplace=True)
    return merged


def build_ccr(df, column='Feedback_Type', limit=10):
    """
    Return a DataFrame of percent CCRs

    Parameters
    ----------
        df (DataFrame): Data with compliments, concerns, and recommendations
        column (str): Name of Feedback Type column.
        normalize (bool): Return counts as percentages. Default is True.
        all_responses (bool): Include no responses and non-responses?
        limit (int): minimum number of respondenses

    Returns
    -------
        DataFrame with percentages for CCRs
    """
    A = df[column].str.startswith('Will not')
    B = df[column].notnull()

    series = df[~A & B][column]

    counts = series.value_counts(normalize=True) * 100
    ccr_df = counts.round(1).reset_index(name='Percent')
    index_ = pd.Index(ccr_df['index'], name=column)
    ccr_df.index = index_
    return ccr_df.drop('index', axis=1)


def build_dn_dp(data, *args, **kwargs):
    """
    Create the `dn` and `dp` datasets from a larger dataset (e.g., `d90`)

    Parameters
    ----------
        data (DataFrame): Main dataset (d90) containing 90 days worth of data
        args (list): Assumed dates are reverse chronological order
        
        kwargs
        ------
            dne: Last date of current period
            dns: First date of current period
            dpe: Last date of previous period
            dps: First date of previous period

    Returns
    -------
        Two DataFrames with data separate into current and previous periods

    Notes
    -----
        Assumed order for *args: (dne, dns, dpe, dps)
    """
    if args:
        if isinstance(args[0], tuple):
            dne, dns, dpe, dps = args[0]
        elif len(args) == 4:
            dne, dns, dpe, dps = args
    elif kwargs:
        dne, dns = kwargs['dne'], kwargs['dns']
        dpe, dps = kwargs['dpe'], kwargs['dps']

    date_column = kwargs.get('date_column', 'ResponseDateTime')
    dn = data[(data[date_column] < dne) & (data[date_column] >= dns)]
    dp = data[(data[date_column] < dpe) & (data[date_column] >= dps)]
    return dn, dp


def build_tot(df):
    """ Creates the pivot table for the `tot` dataset """
    pivot = df.pivot(index='SurveyType', columns='Date', values='Trust')
    pivot.columns = df.label.unique()
    return pivot

"""
Function Group: Functions to normalize and standardize variable values.
E.g., Period of Service, Representative Service Organization
"""


def assign_war_from_datetime(name, wars=None, default="Peacetime"):
    """
    Assign a readable String to a date indicating a period of service.

    Parameters
    ----------
        name (datetime): Period of service date
        wars (list): Pickled file with war information
        default (str): value to assign to `name` if outside of war periods

    Returns
    -------
        Period of service label
    """
    if wars is None:
        wars = load_periods_of_service()

    for war, startdate, enddate in wars:
        if name.year >= startdate and name.year <= enddate:
            return war
    return default


def assign_war_from_string(name, regex=r"\d{2}[\\/]\d{2}[\\/]\d{2,4}"):
    """
    Assign a period of service String to a date String indicating a period of
    service.

    Parameters
    ----------
        name (datetime): Period of service date
        regex (str): regular expression for a datetime input.

    Returns
    -------
        Period of service label
    """
    if pd.notnull(name):

        date_range = re.compile(regex, re.IGNORECASE)
        result = date_range.findall(name)

        # If a date range is found, try to assign that date range to a period
        if len(result) == 2:
            name_startdate, name_enddate = result

            date_1 = date_convert(name_startdate, 1, pat_in="%m/%d/%Y")
            date_2 = date_convert(name_enddate, 1, pat_in="%m/%d/%Y")

            war_1 = assign_war_from_datetime(date_1)
            war_2 = assign_war_from_datetime(date_2)

            if war_1 == war_2:
                return war_1
            else:
                return "Multiple Periods"

        # If there are more than two hits, return `Multiple Periods`
        elif len(result) > 2:
            return "Multiple Periods"
    return name

def normalize_service_name(name, pattern=r"\d{,2} - (\w+ \w+).*"):
    """
    Extract the period of service label from a messy string input.

    Parameters
    ----------
        name (str): Possibly messy string input with period of service label
        pattern (str): regular expression to capture service name

    Returns
    -------
        List of regex results
    """
    regex = re.compile(pattern, re.IGNORECASE)
    return regex.findall(name)

def replace(text):
    """
    Replace non-standard Period of Service information from input text.

    Parameters
    ----------
        text (str): Possible messy text with period of service label

    Returns
    -------
        Normalized period of service label

    """
    found = normalize_service_name(str(text))
    if found:
        return found[0]
    return assign_war_from_string(text)


def standardize_period_of_service(df, col='PeriodOfService'):
    """
    Normalize the names of various periods of service in a DataFrame.

    Parameters
    ----------
        df (DataFrame): Data with period of service information
        col (str): Column name with period of service information

    Returns
    -------
        DataFrame with normalized periods of service
    """
    df.loc[:, col] = df[col].apply(replace).copy()
    return df

# %% Regional Office Normalization
    

def normalize_regional_office(office, pattern=r"\d{,3} - (.+)"):
    """
    Extract just the Regional Office name.

    Parameters
    ----------
        office (str): Possible messy regional office name
        pattern (str): regular expression to extract just the name.

    Returns
    -------
        Shortened name of input regional office
    """
    regex = re.compile(pattern, re.IGNORECASE)
    found = regex.findall(office)
    if found:
        return found[0]
    return office

# %% Representative Standardization
    

def normalize_representative(name, pattern="^(\w) -"):
    """
    Expand acronyms used for representatives.

    Parameters
    ----------
        name (str): Possibly messy Representative label

    Returns
    -------
        Normalized Representative label
    """
    representatives = load_representatives()
    name = name.lower()

    if '-' in name:
        name = name[0]

    if name in representatives:
        return representatives[name]
    return name


def normalize_representatives(df):
    """
    Expand acronyms and normalize the names of representatives.

    Parameters
    ----------
        df (DataFrame): Data containing Representative information

    Returns
    -------
        DataFrame with normalized Representative labels
    """
    F = normalize_representative
    df.loc[:, "Representative"] = df.Representative.apply(F)
    return df

# %% Race Data Standardization


def normalize_race_data(df):
    """ Ensure there is data present and make answers title case. """
    df.loc[:, 'race'] = df.race.apply(lambda x: str(x).title() if x else x)
    try:
        df.rename(columns={'race': 'Race'}, inplace=True)
    except:
        pass
    return df


def normalize_survey_type(df, regex=r"hearing|nod|appeal|decision"):
    """ Ensure all SurveyType names are standardized. """
    regex = re.compile(regex, re.IGNORECASE)
    strip = lambda x: regex.findall(x.strip())[0]
    return df.assign(SurveyType=df.SurveyType.apply(strip))


def normalize(df):
    """
    Normalize values within a DataFrame.

    Parameters
    ----------
        df (str): Data containing the following columns (not exact order):
                       - Period of Service
                       - Representative
                       - SurveyType
                       - Race
                       - Age Group

    Returns
    -------
        DataFrame with normalized columns

    """
    for func in [standardize_period_of_service,
                 normalize_representatives,
                 normalize_survey_type,
                 normalize_race_data,
                 assign_age_groups]:
        try:
            df = func(df)
        except:
            pass
    
    column_mapping = {'PeriodOfService': 'Period of Service',
                      'Age_Group': 'Age Group',
                      'race': 'Race',
                      'gender': 'Gender'}
    df = df.rename(column_mapping, axis=1)
    
    
    columns = ['Period of Service', 'Race', 'Gender']
    for column in columns:
        df.loc[:, column] = df[column].str.strip().str.title()

    mapping = {"M": "Male", "F": "Female"}
    df.loc[:, 'Gender'] = df['Gender'].replace(mapping)
    
    return df

# %% Descriptive Statistics and BLUF Figures
#def as_date_string(func, *args, **kwargs):
#    pattern = kwargs.get('pattern', "%B %d, %Y")
#    def date_wrapper(*args, **kwargs):
#        return f"{func(*args, **kwargs):{pattern}}".replace(" 0", " ")
#    return date_wrapper

#@as_date_string
def report_date(full_date=True, pattern="%B %d, %Y"):
    """ Get today's date (e.g., for report date/time stamping) """
    date = config.today
    return f"{date:{pattern}}".replace(" 0", " ")

#@as_date_string
def first_date(df, column="ResponseDateTime", pattern="%B %d, %Y"):
    """ Get the earliest date a response was submitted in the data """
    date = df[column].min()
    return f"{date_format(date, 1):{pattern}}".replace(" 0", " ")

#@as_date_string
def last_date(df, column="ResponseDateTime", pattern="%B %d, %Y"):
    """ Get the latest date a response was submitted """
    date = df[column].max()
    return f"{date_format(date, 1):{pattern}}".replace(" 0", " ")


def row_count(func):
    """ Wrapper function to count a DataFrame's rows """
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs).shape[0]
    return wrapper


@row_count
def number_of_respondents(df, column='SurveyPersonID'):
    """ Get number of survey responses by SurveyPersonID or row count """
    if column in df.columns:
        return df[column].unique()
    return df.iloc[:,0]


@row_count
def number_of_comments(df):
    """ Get number of free-text comments in a DataFrame. """
    CHARS = df.Feedback_Comment.notnull()
    VALID = df.Feedback_Comment.str.strip().str.len() > 0
    return df[CHARS & VALID]


def number_of_comments_by(df, column, kw):
    """ Get number of comments by group """
    ENTRY = df[column].notnull()
    VALID = df[column] == kw
    return number_of_comments(df[ENTRY & VALID])


def nresp(df, column='SurveyPersonID'):
    """ Alias for number_of_respondents() """
    return number_of_respondents(df, column)


def ncomments(df):
    """ Alias for number_of_comments() """
    return number_of_comments(df)


def number_meets_minimum(df, column='# of Respondents', minimum=10):
    """
    Check element-wise if the number of respondents meets the minimal threshold

    Parameters
    ----------
        df (DataFrame): data containing respondent counts
        column (str): name of column containing respondent counts
        minimum (int): minimal threshold for a value to be True, else False

    Returns
    -------
        Series of boolean values
    """
    return df[column] >= minimum

"""
Function Group: percent_*

Description: Percentage Trust and Percentage CCR Functions

Functions for Percent CCR:
    1. percent_ccr: Percentages of compliments, concerns, recommendations
    2. percent_*: Replace * with compliment, concern, or recommendation

Functions for Percent Trust:
    3. percent_trust: Overall percent Trust of input data set
    4. percent_trust_by: Percent Trust of input data set by value in column
    5. percent_trust_*: Replace * with AMA, Legacy, Hearing, or Decision
"""
# %% Functions to get Percentages


def percent_ccr(df, feedback_type='Feedback_Type', column='Percent'):
    """
    Get proportion of compliments, concerns and recommendations.

    Parameters
    ----------
        df (DataFrame): Table with compliments, concerns, and recommendations
        feedback_type (str): column name containing feedback types
        column (str): Column name with percentage values

    Returns
    -------
        Float indicating proportion of surveys with `feedback_type`
    """
    return df.loc[feedback_type, column].copy()


def percent_compliment(df, key='Compliment'):
    """ Get the proportion of Feedback that was a Compliment """
    return df.loc[key, ].Percent


def percent_concern(df, key='Concern'):
    """ Get the proportion of Feedback that was a Concern """
    return df.loc[key, ].Percent


def percent_recommendation(df, key='Recommendation'):
    """ Get the proportion of Feedback that was a Recommendation """
    return df.loc[key, ].Percent


def percent_trust(df, column='Feedback_Type'):
    """ Get the Trust score rounded to one significant digit """
    trust = calculate_trust(df)
    if trust is not None:
        return round(trust, 1)
    return None


def percent_trust_by(df, column, key):
    """ Get the Trust score for a column by value """
    NOT_NULL = df[column].notnull()
    IS_VALID = df[column].isin([key])
    return percent_trust(df[NOT_NULL & IS_VALID])


def percent_trust_AMA(df, column='SurveyType', key='NOD'):
    """ Get the Trust score for the AMA survey """
    return percent_trust_by(df, column, key)


def percent_trust_legacy(df, column='SurveyType', key='Appeal'):
    """ Get the Trust score for the Legacy survey """
    return percent_trust_by(df, column, key)


def percent_trust_hearing(df, column='SurveyType', key='Hearing'):
    """ Get the Trust score for the Hearing survey """
    return percent_trust_by(df, column, key)


def percent_trust_decision(df, column='SurveyType', key='Decision'):
    """ Get the Trust score for the Decision survey """
    return percent_trust_by(df, column, key)

# %% Function to grab variable value name with most/least Trust


def index_trust(table, func='max'):
    """ Return the table row corresponding to max/min Trust score. """
    return table[table.Trust == table.Trust.agg([func])[0]]


def most_trust(table):
    """ Return the category corresponding to the highest Trust score. """
    return index_trust(table).iloc[0, 0]


def least_trust(table):
    """ Return the category corresponding to the lowest Trust score. """
    return index_trust(table, 'min').iloc[0, 0]


def most_trust_count(table):
    """ Return the number of respondents with the highest Trust score. """
    return index_trust(table).iloc[0, 1]


def least_trust_count(table):
    """ Return the number of respondents with the lowest Trust score. """
    return index_trust(table, 'min').iloc[0, 1]


def most_trust_value(table):
    """ Return the Trust value equal to the highest Trust score. """
    return index_trust(table).iloc[0, 2]


def least_trust_value(table):
    """ Return the Trust value equal to the lowest Trust score. """
    return index_trust(table, 'min').iloc[0, 2]


def most_trust_appeal(table, reverse=False):
    """
    Return the appeals Survey label with the higher Trust score.

    Parameters
    ----------
        table (DataFrame): Data with counts and Trust scores by SurveyType
        reverse (bool): Return the Appeal process with least Trust?
    """
    ama = percent_trust_AMA(table)
    leg = percent_trust_legacy(table)

    FIRST = config.lists.board_surveys[0]
    SECOND = config.lists.board_surveys[1][1]

    if reverse:
        FIRST, SECOND = SECOND, FIRST

    if ama is not None and leg is not None:
        if ama >= leg:
            return FIRST
        return SECOND
    elif ama is not None:
        return FIRST
    elif leg is not None:
        return SECOND
    else:
        return None


def least_trust_appeal(table):
    """ Return the Trust value equal to the lowest Trust score. """
    return most_trust_appeal(table, True)


def hilo(table1, table2):
    """
    Return whether or not AMA surveys are doing better this week.

    Parameters
    ----------
        table1 (DataFrame): Data with counts and Trust scores by SurveyType
        table2 (DataFrame): Data with counts and Trust scores by SurveyType

    Returns
    -------
        Label stating if the table1 has higher or lower trust with regard to
        the NOD / AMA survey. If there is a None (null) value returned, then
        the response will be `not calculable (no responses)`
    """

#    HI, LO = "showed higher", "showed lower"
#    NA = "were insufficient to calculate"

    ama_current = percent_trust_AMA(table1)
    ama_before = percent_trust_AMA(table2)

    if ama_current and ama_before:
        difference = ama_current - ama_before
        if difference > 0:
            return config.bluf.hilo.higher
        return config.bluf.hilo.lower
    return config.bluf.hilo.insufficient
#            return HI
#        return LO
#    return NA


def generate_main_bluf(dn, dp):
    """ 
    Return a BLUF string to summarize the first page of the report 
    
    Parameters
    ----------
        dn (DataFrame): Current time period's data
        dp (DataFrame): Previous time period's data
    
    Returns
    -------
        Custom BLUF string based on the data
    
    Notes
    -----
        BLUF strings can be edited in the configuration.yaml file
    """
    ama_curr = percent_trust_AMA(dn)
    ama_prev = percent_trust_AMA(dp)

    before, current = prev(dn, dp), now(dn, dp)

    if ama_prev is None and ama_curr is None:
        A = config.bluf.ama.no_responses
    elif ama_prev is None or (ama_prev < ama_curr):
        A = config.bluf.ama.higher_trust.format(current=current)
    elif ama_curr is None or (ama_prev > ama_curr):
        A = config.bluf.ama.lower_trust.format(current=current)
    else:
        A = config.bluf.ama.no_change

    leg_curr = percent_trust_legacy(dn)
    if ama_curr is None:
        if leg_curr is None or (leg_curr == 0):
            B = config.bluf.ama_leg.no_responses

        elif leg_curr > 0:
            B = config.bluf.ama_leg.higher_trust_no_ama
            
    else:
        if leg_curr is None:
            B = config.bluf.ama_leg.higher_trust_no_leg
        elif leg_curr < ama_curr:
            B = config.bluf.ama_leg.higher_trust_ama
        elif leg_curr > ama_curr:
            B = config.bluf.ama_leg.higher_trust_leg
        else:
            B = config.bluf.ama_leg.same_trust

    contents = dict(current=current, before=before, A=A, B=B)
    return config.bluf.autobluf.format(**contents)


"""
Function Group: delta_*

Delta (difference) functions that calculate differences in percentages, dates,
and times between two datasets.

Functions for calculating differences between DataFrames:
    1. delta_trust: Calculate the difference in trust between two datasets
    2. delta: Calculate the difference in percents/counts between two datasets
    3. delta_*: Replace * with AMA, legacy, hearing, decision for trust deltas
    4. delta_*: Replace * with compliment, concern, recommendation for % deltas
    5. delta_*: Replace * with respondents or comments for count deltas

"""


def delta_ccr(current, before, kw, metric):
    """ 
    Get the +/- difference for a specified SurveyType. 
    
    Parameters
    ----------
        current (DataFrame): Data for the current time period
        before (DataFrame): Data for the previous time period
        column (str): Name of column with categories to compare
        kw (str): Category to compare between current and before datasets
        metric (str): Type of data to return (e.g., percent, Trust)
    
    Notes
    -----
        These functions are used for the front page of the Board report
    """
    difference = (current - before).loc[kw, metric]
    delta_ = round(difference, 1)
    if delta_ >= 0:
        return f"+{delta_}%"
    return f"{delta_}%"


def delta(current, before, column, kw, metric):
    """ 
    Get the +/- difference for a specified SurveyType. 
    
    Parameters
    ----------
        current (DataFrame): Data for the current time period
        before (DataFrame): Data for the previous time period
        column (str): Name of column with categories to compare
        kw (str): Category to compare between current and before datasets
        metric (str): Type of data to return (e.g., percent, Trust)
    
    Notes
    -----
        These functions are used for the front page of the Board report
    """
    combined = current.merge(before, on=column, how='outer')
    combined.replace(np.nan, 0, inplace=True)
    difference = combined[metric + "_x"]-combined[metric + "_y"]
    combined = combined.assign(delta_=difference)

    try:
        HAS_KW = combined[column] == kw
        delta_ = combined.loc[HAS_KW, 'delta_'].ravel()[0]
        delta_ = round(delta_, 1)
    except(IndexError):
        return None
    if delta_ >= 0:
        return f"+{delta_}%"
    return f"{delta_}%"


def delta_AMA(current, before):
    return delta(current, before, 'SurveyType', 'NOD', 'Trust')


def delta_legacy(current, before):
    return delta(current, before, 'SurveyType', 'Appeal', 'Trust')


def delta_hearing(current, before):
    return delta(current, before, 'SurveyType', 'Hearing', 'Trust')


def delta_decision(current, before, kw='Decision', metric='Trust'):
    return delta(current, before, 'SurveyType', kw, metric)


def delta_compliment(current, before, kw='Compliment', metric='Percent'):
    return delta_ccr(current, before, kw, metric)


def delta_concern(current, before, kw='Concern', metric='Percent'):
    return delta_ccr(current, before, kw, metric)


def delta_recommendation(current, before, kw='Recommendation', metric='Percent'):
    return delta_ccr(current, before, kw, metric)


def delta_respondents(current, before):
    difference = number_of_respondents(current) - number_of_respondents(before)
    if difference >= 0:
        return f"+{difference}"
    return f"{difference}"


def delta_comments(current, before):
    return number_of_comments(current) - number_of_comments(before)


def month(df=None, date=config.today):
    if df is not None:
        date = date_convert(last_date(df), asdatetime=True, pat_in='%B %d, %Y')
    return f"{date:%B}"


def year(df=None, date=config.today):
    if df is not None:
        date = date_convert(last_date(df), asdatetime=True, pat_in='%B %d, %Y')
    return f"{date:%Y}"


def week(df=None, date=config.today):
    if df is not None:
        date = date_convert(last_date(df), asdatetime=True, pat_in='%B %d, %Y')
    return f"{date:%m/%d/%Y}"


def day_difference(time1, time2):
    """ Return the difference in days between two datasets' last days """
    if isinstance(time1, str) & isinstance(time2, str):
        time1 = date_convert(time1, asdatetime=True, pat_in=config.format.iso)
        time2 = date_convert(time2, asdatetime=True, pat_in=config.format.iso)
    return (time1 - time2).days


def report_period(current, before):
    """ Return a time period label """
    days = day_difference(current.ResponseDateTime.max(),
                          before.ResponseDateTime.max())
    if days <= 10:
        return "Week"
    elif days < 28:
        return "Mid-Month"
    return "Month"


def now(current, before):
    """ Return a string stating this period. """
    days = day_difference(current.ResponseDateTime.min(),
                          before.ResponseDateTime.min())
    if days <= 7:
        return "this week"
    elif days <= 14:
        return "these two weeks"
    return "this month"


def prev(current, before):
    """ Return a string stating the previous period. """
    days = day_difference(current.ResponseDateTime.min(),
                          before.ResponseDateTime.min())
    if days <= 7:
        return "last week"
    elif days <= 14:
        return "last two weeks"
    return "last month"


"""
Functions for getting time periods. These functions follow the format: date_*

    1. date_period <- Generic function requiring start and end periods
    2. date_90 <- Get a 90 day period
    3. date_day <- Get a 1 day period
    4. date_week <- get a 1 week period
    5. date_month <- get a 1 month period
"""


def date_string_today():
    return f"{config.today:%B %d, %Y}".replace(" 0", " ")


def today(dt=None):
    """ Return today's date in tuple form """
    if not isinstance(dt, datetime):
        return datetime.today().timetuple()[:3]
    return dt.timetuple()[:3]


def date_convert(dt,
                 asdatetime=False,
                 astuple=False,
                 pat_in="%b '%y",
                 pat_out="%Y-%m-%d"):
    """
    Convert a non-iso date `Jan '01'` to an iso date `2001-01-01`

    Parameters
    ----------
        dt (str): Date string to convert into another date type
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        pat_in (str): Pattern of input date
        pat_out (str): Pattern of output date

    Returns
    -------
        Newly formated date as a string, datetime or tuple object

    Notes
    -----
        Common pattern tags for the example date `January 1, 2001` are:
            year    (%Y, %y)       2001, 01
            month   (%m, %B, %b)   01, January, Jan
            day     (%d)           01

    Examples
    --------
        >>> # Default pattern is "%b '%y"
        >>> date_convert("Feb '19")
        >>> "2019-02-01"
        >>>
        >>> date_convert("Feb 2019", pat_in="%b %Y")
        >>> "2019-02-01"
        >>>
        >>> date_convert("December 25, 2019", pat_in="%B %d, %Y")
        >>> "2019-12-25"
        >>>
        >>> date_convert("2019-06-21", pat_in="%Y-%m-%d")
        >>> "2019-06-21"
        >>>
        >>> date_convert("06/21/19", pat_in="%m/%d/%y")
        >>> "2019-06-21"
        >>>
        >>> date_convert("06/21/19", pat_in="%m/%d/%y", pat_out="%d/%m/%Y")
        >>> "21/06/2019"
        >>>
    """
    dt = datetime.strptime(dt, pat_in)
    if asdatetime or astuple:
        return date_format(f"{dt:%Y-%m-%d}", asdatetime, astuple)
    dt = date_format(f"{dt:%Y-%m-%d}", asdatetime=True)
    return f"{dt:{pat_out}}".replace(" 0", " ")


def date_is_valid(dt):
    """
    Determine if input date is possible or valid

    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format

    Returns
    -------
        True if date is valid, else False
    """
    try:
        date_format(dt)
    except ValueError:
        return False
    else:
        return True
    return False


def date_format(dt,
                asdatetime=False,
                astuple=False,
                astimestamp=False,
                asB=False,
                asW=False,
                next_day=False,
                previous_day=False,
                month=False,
                year=False):
    """
    Format a datetime input (e.g., str, int, tuple, datetime) as another.

    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asB (bool): Return a string formatted as %B %d, %Y
        asW (bool): Return a string formatted as %m/%d
        next_day (bool): Return the date advanced 1 day
        previous_day (bool): Return the date retreated 1 day
        month (bool): Return just the name of the month if True
        year (bool): Return just the year if True

    Returns
    -------
        Datetime or tuple depending on flag. Datetime flag takes priority
        over tuple if both are True. Function returns an ISO string if both
        are False.
    """
    if dt is None:
        return None

    IS_STRING = isinstance(dt, str)
    IS_TUPLE = isinstance(dt, tuple)
    IS_INTEGER = isinstance(dt, int)

    DATETYPES = [date, datetime, pd.Timestamp]
    IS_DATETIME = any(isinstance(dt, datetype) for datetype in DATETYPES)

    # Determine incoming date format
    if IS_STRING:
        dt = pd.Timestamp(dt)

    elif IS_TUPLE:

        # Resolve a day that hangs over from a month difference
        _year, _month, _day = dt
        if calendar.mdays[_month] < _day:
            _month += 1
            _day = 1
        dt = (_year, _month, _day)
        dt = date(*dt)

    elif IS_DATETIME:
        dt = date(*dt.timetuple()[:3])

    elif IS_INTEGER:
        dt = date.fromordinal(dt)

    else:
        warning = "'dt' needs to be a str, tuple, int, or datetime type"
        raise AttributeError(warning)

    # Standardize datetime as a Timestamp
    dt = pd.Timestamp(dt)

    # Advance or regress the date by one day
    if next_day:
        dt = dt + pd.Timedelta(days=1)

    if previous_day:
        dt = dt - pd.Timedelta(days=1)
        
    # Output the date according to the format indicated
    if asdatetime:
        return dt
    elif astuple:
        return dt.timetuple()[:3]
    elif asB:
        return f"{dt:%B %d, %Y}".replace(' 0', ' ')
    elif asW:
        return f"{dt:%m/%d}".replace(' 0', ' ')
    elif month:
        return f"{dt:%B}"
    elif year:
        return f"'{dt:%y}"
    else:
        timetuple = dt.timetuple()[:3]
        return date(*timetuple).isoformat()


def date_delta(dt=None,
               years=0,
               months=0,
               days=0,
               asdatetime=False,
               astuple=False,
               astimestamp=False,
               asstring=False,
               adjust=True):
    """
    Return a tuple of two dates (end, start). Assumes input date is last date.

    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        years (int): Number of years to subtract from dt
        months (int): Number of months to subtract from dt
        days (int): Number of days to subtract from dt
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asstring (bool): Return an isoformatted string?
        adjust (bool): Account for Pacific time submission (e.g., -3 hrs)?

    Returns
    -------
        Tuple of datetimes or tuples depending on flag parameters.
        Datetime flag takes priority over tuple if both are True.
        Function returns ISO strings if both are False.
    """
    if dt is None:
        dt = today(dt)

    YEAR, MONTH, DAY = date_format(dt, astuple=True)

    year = YEAR - years
    month = MONTH - months

    if month < 1:
        year -= 1
        month += 12

    if adjust:
        days += 1
        dt = date_format(dt, 1) - timedelta(1)

    nt = date_format((year, month, DAY), 1) - timedelta(days)

    params = dict(asdatetime=asdatetime,
                  astuple=astuple,
                  astimestamp=astimestamp)
    date_one = date_format(dt, **params)
    date_two = date_format(nt, **params)

    if any(params.values()):
        return date_one, date_two
    elif asstring:
        return date_format(date_one), date_format(date_two)
    return f"{date_one} 21:00:00", f"{date_two} 21:00:00"


def date_ranges(**kwargs):
    """
    Return four dates representing the start and end of two time ranges

    Parameters (kwargs)
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        years (int): Number of years to subtract from dt
        months (int): Number of months to subtract from dt
        days (int): Number of days to subtract from dt
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asstring (bool): Return an isoformatted string?
        adjust (bool): Account for Pacific time submission (e.g., -3 hrs)?

    Returns
    -------
        Tuple of four dates as string, datetime, tuple, or Timestamp
    """
    dt = date_delta(**kwargs)
    kwargs.update({'dt': dt[1], 'adjust': False})
    pt = date_delta(**kwargs)
    return (*dt, *pt)


def date_120(dt=None,
             asdatetime=False,
             astuple=False,
             astimestamp=False,
             asstring=False,
             adjust=True):
    """
    Return the first and last dates of the previous 120 days
    
    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asstring (bool): Return an isoformatted string?
        adjust (bool): Account for Pacific time submission (e.g., -3 hrs)?
    
    Notes
    -----
        Parameters are used for date_delta()
    """
    return date_delta(days=120, **locals())


def date_90(dt=None,
            asdatetime=False,
            astuple=False,
            astimestamp=False,
            asstring=False,
            adjust=True):
    """
    Return the first and last dates of the previous 90 days
    
    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asstring (bool): Return an isoformatted string?
        adjust (bool): Account for Pacific time submission (e.g., -3 hrs)?
    
    Notes
    -----
        Parameters are used for date_delta()
    """
    return date_delta(days=90, **locals())


def date_day(dt=None,
             asdatetime=False,
             astuple=False,
             astimestamp=False,
             asstring=False,
             adjust=True):
    """
    Return the first and last dates of the current day
    
    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asstring (bool): Return an isoformatted string?
        adjust (bool): Account for Pacific time submission (e.g., -3 hrs)?
    
    Notes
    -----
        Parameters are used for date_delta()
    """
    return date_delta(days=1, **locals())


def date_week(dt=None,
              asdatetime=False,
              astuple=False,
              astimestamp=False,
              asstring=False,
              adjust=True):
    """
    Return the first and last dates of the current and previous week
    
    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asstring (bool): Return an isoformatted string?
        adjust (bool): Account for Pacific time submission (e.g., -3 hrs)?
    
    Notes
    -----
        Parameters are used for date_ranges()
    """
    days = 7
    return date_ranges(**locals())


def date_fortnight(dt=None,
                   asdatetime=False,
                   astuple=False,
                   astimestamp=False,
                   asstring=False,
                   adjust=True):
    """
    Return the first and last dates of the current and previous month
    
    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asstring (bool): Return an isoformatted string?
        adjust (bool): Account for Pacific time submission (e.g., -3 hrs)?
    
    Notes
    -----
        Parameters are used for date_ranges()
    """
    days = 14
    return date_ranges(**locals())


def date_month(dt=None,
               asdatetime=False,
               astuple=False,
               astimestamp=False,
               asstring=False,
               adjust=True):
    """
    Return the first and last dates of the current and previous month
    
    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asstring (bool): Return an isoformatted string?
        adjust (bool): Account for Pacific time submission (e.g., -3 hrs)?
    
    Notes
    -----
        Parameters are used for date_delta() and date_ranges()
    """
    months = 1
    locals_ = locals()
    if not dt:
        year, month, day = date_format(config.today, astuple=True)
        locals_['dt'] = date_format((year, month, 1))
    return date_ranges(**locals_)


def date_midmonth(dt=None,
                  asdatetime=False,
                  astuple=False,
                  astimestamp=False,
                  asstring=False,
                  adjust=True):
    """
    Return the first and last dates of the midmonth and previous month
    
    Parameters
    ----------
        dt (str, tuple, int, datetime): Date or time in ISO format
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?
        astimestamp (bool): Return a pandas Timestamp?
        asstring (bool): Return an isoformatted string?
        adjust (bool): Account for Pacific time submission (e.g., -3 hrs)?
    
    Notes
    -----
        Parameters are used for date_delta()
    """
    dt = date_format(dt, asdatetime=True)
    days = dt.day - 1
    locals_1 = locals()
    now = date_delta(**locals_1)
    locals_1.pop('days')
    locals_1.update({'dt': now[1], 'months': 1, 'adjust': False})
    pre = date_delta(**locals_1)
    return (*now, *pre)


def date_quarter(q, year=config.today.year, asdatetime=False, astuple=False):
    """
    Return the first and last months of a specified quarter according to the
    federal fiscal year

    Parameters
    ----------
        q (int): Quarter number
        year (int): Reference year (e.g., current year)
        asdatetime (bool): Return a datetime object?
        astuple (bool): Return a tuple?

    Returns
    -------
        Datetime or tuple depending on flag. Datetime flag takes priority
        over tuple if both are True. Function returns an ISO string if both
        are False.
    """
    quarters = {1: [10, 11, 12],
                2: [1, 2, 3],
                3: [4, 5, 6],
                4: [7, 8, 9]}
    
    if not isinstance(q, int):
        try:
            q = int(q)
        except ValueError:
            raise ValueError("Please enter a valid quarter (1, 2, 3, 4)")
    elif q > 4:
        raise ValueError("Please enter a valid quarter (1, 2, 3, 4)")

    month1, __, month2 = quarters[q]
    start = (year - 1 if month1 > 9 else year, month1, 1)
    end = (year, 1 if month2 == 12 else month2, 1)
    date_one = date_format(start, asdatetime, astuple)
    date_two = date_format(end, asdatetime, astuple)
    return date_one, date_two


def date_Q1(asdatetime=False, astuple=False):
    """ Return the first and last dates of the first fiscal quarter """
    return date_quarter(1, asdatetime=asdatetime, astuple=astuple)


def date_Q2(asdatetime=False, astuple=False):
    """ Return the first and last dates of the second fiscal quarter """
    return date_quarter(2, asdatetime=asdatetime, astuple=astuple)


def date_Q3(asdatetime=False, astuple=False):
    """ Return the first and last dates of the third fiscal quarter """
    return date_quarter(3, asdatetime=asdatetime, astuple=astuple)


def date_Q4(asdatetime=False, astuple=False):
    """ Return the first and last dates of the fourth fiscal quarter """
    return date_quarter(4, asdatetime=asdatetime, astuple=astuple)


def date_ytd(dt=config.today, asdatetime=False, astuple=False):
    """ Return the Board launch date and today's date """
    launch_date = (2018, 10, 16)
    date_one = date_format(launch_date, asdatetime=asdatetime, astuple=astuple)
    date_two = date_format(dt, asdatetime=asdatetime, astuple=astuple)
    return date_one, date_two


"""
Survey functions to process VSignals Survey Distribution data (Sattrackdist).

UPDATE SEPTEMBER 26, 2019
QUESTIONS ARE NOW RETRIEVED FROM OUR INTERNAL DATABASES AND THIS SATTRACKDIST
SECTION WILL BE REMOVED.

These functions are necessary because the Sattrackdist "STD" data from
VSignals is irregularly structured for computing purposes.

If the STD data format from VSignals changes, these functions should become
deprecated.

Prefix:
    survey_*
"""


def survey_agree(df, n=0, dummy=None):
    """
    Get the top n questions with most `Agree` answers

    Parameters
    ----------
        df (DataFrame): Full data set result from pull (e.g., d90)
        n (int): Rank of question to return. Default is 0 (most agreement)
        dummy (None): Placeholder variable for paradigm

    Returns
    -------
        Series of question ranked n in most `Agree` answers
    """
    questions = calculate_trust_by_question(df, config.lists.agree_scores)
    sorted_questions = questions.sort_values('Score', ascending=False)

    try:
        question = sorted_questions.iloc[n]
        question.name = question.Survey
    except IndexError:
        question = None
    return question

def survey_disagree(df, n=0, dummy=None):
    """
    Get the top n questions with most `Disagree` answers

    Parameters
    ----------
        df (DataFrame): Full data set result from pull (e.g., d90)
        n (int): Rank of question to return. Default is 0 (most disagreement)
        dummy (None): Placeholder variable for paradigm

    Returns
    -------
        Series of question ranked n in most `Disagree` answers
    """
    questions = calculate_trust_by_question(df, config.lists.disagree_scores)
    sorted_questions = questions.sort_values('Score', ascending=False)

    try:
        question = sorted_questions.iloc[n]
        question.name = question.Survey
    except IndexError:
        question = None
    return question

def survey_agreement_by_survey(df, *args, **kwargs):
    """
    Get questions and Trust scores with the most `Disagree` answers by Survey

    Parameters
    ----------
        df (DataFrame): Full data set result from pull (e.g., d90)
        columns (list): Column headers to return

    Returns
    -------
        DataFrame of questions with most `Disagree` answers by survey
    """
    columns = kwargs.get('columns', ['Survey', 'FriendlyWording', 'Score'])
    scores = kwargs.get('scores', config.lists.disagree_scores)

    questions = calculate_trust_by_question(df, scores=scores)
    questions = questions.sort_values(['Survey', 'Score'], ascending=False)
    question = questions.groupby('Survey').FriendlyWording.first()
    question = question.reset_index()
    question.columns = columns[:2]

    merged = question.merge(questions[columns], on=columns[:2])
    merged.index = merged.Survey
    return merged

def survey_isolate(df, survey, best=False):
    """
    Return Decision questions from VSignals data.

    Parameters
    ----------
        df (DataFrame): Full data set result from pull (e.g., d90)
        survey (str): Name of Board survey to isolate
        best (bool): isolate best performing (True) or worst performing (False)

    Returns
    -------
        Board survey data as a Series
    """
    scores = config.lists.agree_scores if best else config.lists.disagree_scores
    surveys = survey_agreement_by_survey(df, scores=scores)
    current = surveys.loc[survey]
    if current.Score == 0 or pd.isnull(current.Score):
        return None
    return current

def survey_isolate_question(df, n=0, best=False):
    """
    Return question with highest (best=True) or lowest (best=False) scores.

    Parameters
    ----------
        df (DataFrame): Full data set result from pull (e.g., d90)
        n (int): Question rank to return. Default is 0.
        best (bool): isolate best performing (True) or worst performing (False)

    Returns
    -------
        Question string.

    Notes
    -----
        As of 2019-11-21 the input `df` requires input from either the
        survey_agree or survey_disagree feeder functions.
    """
    return df['FriendlyWording']
#        return survey_agree(df, n)['FriendlyWording']
#    else:
#        return survey_disagree(df, n)['FriendlyWording']

def survey_isolate_score(df, n=0, best=False):
    """
    Return the highest (best=True) or lowest (best=False) scores.

    Parameters
    ----------
        df (DataFrame): Full data set result from pull (e.g., d90)
        n (int): Question rank to return. Default is 0.
        best (bool): isolate best performing (True) or worst performing (False)

    Returns
    -------
        Question string.

    Notes
    -----
        As of 2019-11-21 the input `df` requires input from either the
        survey_agree or survey_disagree feeder functions.
    """
    return df['Score']
#    if best:
#        return survey_agree(df, n)['Score']
#    else:
#        return survey_disagree(df, n)['Score']

def survey_isolate_survey(df, n=0, best=False):
    """
    Return Survey with highest (best=True) or lowest (best=False) scores.

    Parameters
    ----------
        df (DataFrame): Full data set result from pull (e.g., d90)
        n (int): Question rank to return. Default is 0.
        best (bool): isolate best performing (True) or worst performing (False)

    Returns
    -------
        Question string.

    Notes
    -----
        As of 2019-11-21 the input `df` requires input from either the
        survey_agree or survey_disagree feeder functions.
    """
    return df['Survey']
#    if best:
#        return survey_agree(df, n)['Survey']
#    else:
#        return survey_disagree(df, n)['Survey']


def survey_top(questions, n=0, formatted=True):
    """
    Return the worst-performing question from Sattrackdist data. If there
    is no worst-performing question, return N/A.

    Parameters
    ----------
        questions (DataFrame): data with Survey, FriendlyWording, and Trust
        n (int): Question rank to return

    Returns
    -------
        Top n question as a string
    """
    NO_DATA_RESPONSE = "N/A (Insufficient data available)"
    BVA_SURVEY_QUESTION = ftfy.fix_text(config.lists.bva_survey_question[0])

    try:
        if isinstance(questions, pd.DataFrame):
            current = questions.iloc[n]
        else:
            current = questions

        question_text = {"question": current.FriendlyWording,
                         "survey": current.name,
                         "score": round(current.Score, 1)}
        if formatted:
            return BVA_SURVEY_QUESTION.format(**question_text)
        return question_text.values()
    
    except (IndexError, AttributeError):
        return NO_DATA_RESPONSE


"""
A-11 CX Domain functions to find top and bottom drivers

Prefix:
    a11_*

"""


def a11_most_label(domains, column='Score', drop='Confidence_Trust'):
    """
    Get the A11-domain with the most Trust (excl. Confidence_Trust driver)

    Parameters
    ----------
        domains (pd.Series): each domain and its Trust score
        column (str): column name with percentage agreement score
        drop (str): column name with Confidence_Trust driver to drop

    Returns
    -------
        A-11 CX domain label with most Trust
    """
    current = domains.sort_values(column, ascending=False)
    current.drop(drop, inplace=True, errors='ignore')
    domain_most_trust = current.iloc[0].name
    return domain_most_trust.replace("_", "/")


def a11_least_label(domains, column='Score', drop='Confidence_Trust'):
    """
    Get the A11-domain with the least Trust (excl. Confidence_Trust driver)

    Parameters
    ----------
        domains (pd.Series): each domain and its Trust score
        column (str): column name with percentage agreement score
        drop (str): column name with Confidence_Trust driver to drop

    Returns
    -------
        A-11 CX domain label with least Trust
    """
    current = domains.sort_values(column, ascending=False)
    current.drop(drop, inplace=True, errors='ignore')
    domain_least_trust = current.iloc[-1].name
    return domain_least_trust.replace("_", "/")


def a11_most(domains, column='Score', drop='Confidence_Trust'):
    """
    Get the A11-domain with the most Trust (excl. Confidence_Trust driver)

    Parameters
    ----------
        domains (pd.Series): each domain and its Trust score
        column (str): column name with percentage agreement score
        drop (str): column name with Confidence_Trust driver to drop

    Returns
    -------
        A-11 CX domain label with most Trust
    """
    current = domains.sort_values(column, ascending=False)
    current.drop(drop, inplace=True, errors='ignore')
    return current.iloc[0].Score



def a11_least(domains, column='Score', drop='Confidence_Trust'):
    """
    Get the A11-domain with the least Trust (excl. Confidence_Trust driver)

    Parameters
    ----------
        domains (pd.Series): each domain and its Trust score
        column (str): column name with percentage agreement score
        drop (str): column name with Confidence_Trust driver to drop

    Returns
    -------
        A-11 CX domain label with least Trust
    """
    current = domains.sort_values(column, ascending=False)
    current.drop(drop, inplace=True, errors='ignore')
    return current.iloc[-1].Score



def a11_trust(domains, column='Confidence_Trust'):
    """
    Get the Confidence/Trust A11-driver score

    Parameters
    ----------
        domains (pd.Series): each domain and its Trust score
        column (str): column name with Confidence_Trust score

    Returns
    -------
        Confidence/Trust score
    """
    current = domains.loc[column]
    return current.Score



def a11_image(domains, output=config.paths.a11_image):
    """
    Insert the A-11 CX Drivers chart from the ./images folder.

    Parameters
    ----------
        domains (Series): A-11 CX Domains and Trust scores
        output (str): path to image file
    """
    canvas.saveplot_A11(domains, output=output)
    return output

"""
Trust over Time (tot) by SurveyType functions to find top and bottom drivers

Prefix:
    tot_*

"""


def tot_image(tot, output=config.paths.tot_image):
    """
    Insert the Trust over Time (TOT) chart from the ./images folder.

    Parameters
    ----------
        tot (DataFrame): Trust over Time data
        output (str): path to image file
        
    Notes
    -----
        Requires canvas.py script
    """
    canvas.saveplot_TOT(tot, output=output)
    return output

"""
CCR functions to find top compliments, concerns, and their associated comments.

Prefix:
    ccr_*

"""
#CCR_COLUMNS = ['Feedback_Comment', 'Feedback_Type', 'RO', 'ResponseDateTime']
#CCR_FEATURES= ['FeatureReference', 'Contents']
#CCR_PRETTIFY_FEATURES = ['FeatureReference', 'Reference4Presentation']


def ccr_filter_columns(df, columns=config.lists.ccr_columns, limit=5):
    """
    Reduce the dataset by filtering out Feedback/Comments with fewer than
    the number of tokens specified.

    Parameters
    ----------
        df (DataFrame): Feedback comment data
        columns (list): Columns related to Feedback/Comments
        limit (int): Minimum number of tokens to filter out comments by.

    Return
    ------
        Filtered DataFrame
    """
    sub_df = df.filter(items=columns)[df.Feedback_Comment.notnull()].copy()
    sub_df.ResponseDateTime = pd.to_datetime(sub_df.ResponseDateTime)

    tokens = sub_df.Feedback_Comment.str.split().str.len()
    sub_df = sub_df.assign(tokens=tokens)

    sub_df = sub_df.loc[sub_df.tokens >= limit].copy()
    sub_df.sort_values(['tokens'], inplace=True)

    sub_df.reset_index(drop=True, inplace=True)
    return sub_df


def ccr_get_topics(topics=config.paths.topics, features=config.lists.ccr_features):
    """
    Yield a tuple of topic names and their associated regular expressions.

    Parameters
    ----------
        topics (str): Filepath to a dataset of topics, feature_names, and regex
        features (list): Column names to subset `topics` dataset.

    Yields
    ------
        Tuple of `pretty` feature name and feature regular expression
    """
    topics = load_topics(topics)

    for i, (feature_reference, regex) in topics[features].iterrows():
        yield(feature_reference, re.compile(regex, re.IGNORECASE))


def ccr_augment(df):
    """
    Add extra columns for each topic considered.

    Parameters
    ----------
        df (DataFrame): Comments data

    Returns
    -------
        DataFrame with extract columns filled with zeros for each topic
    """
    df = ccr_filter_columns(df)

    # First get the number of topics and feature names
    ntopics = 0
    features = []

    for feature_reference, __regex in ccr_get_topics():
        ntopics += 1
        features.append(feature_reference)

    # Define the size of the addition
    shape = (df.shape[0], ntopics)
    zeros = np.zeros(shape=shape)

    augmented = pd.DataFrame(zeros, columns=features)
    return pd.concat([df, augmented], axis=1)


def ccr_detect_topics(df):
    """
    Insert topic counts into an augmented DataFrame.

    Parameters
    ----------
        df (DataFrame): Augmented data (see ccr_augment() method)

    Returns
    -------
        Topics DataFrame
    """
    features = [f for f, t in ccr_get_topics()]

    # Make sure that the dataset is augmented
    if len(set(features)) != len(set(features).intersection(df.columns)):
        df = ccr_augment(df)

    for i, row in df.iterrows():
        if pd.notnull(row.Feedback_Comment):
            occurrences = 0

            for feature, regex in ccr_get_topics():
                if regex.findall(row.Feedback_Comment):
                    occurrences += 1
                    df.loc[i, feature] = occurrences
    return df


def ccr_list_topics(df, feedback_type='Compliment'):
    """
    Get top compliments and concerns from an augmented DataFrame.

    Parameters
    ----------
        df (DataFrame) Augmented DataFrame with topic counts.
        feedback_type (str): Type of feedback to filter df by.

    Returns
    -------
        Sorted topics by proportion
    """
    features = [f for f, t in ccr_get_topics()]

    filtered = df.loc[df.Feedback_Type==feedback_type].copy()

    feedback = filtered[features]
    feedback = (feedback.sum() / feedback.sum().sum() * 100.)
    return feedback.sort_values(ascending=False)


def ccr_prettify(feedback, features=config.lists.ccr_features_prettify, prettify=True):
    """
    Return top compliments and concerns as readable strings.
    """
    for feature, reference in ccr_get_topics(features=features):
        if feature in feedback:
            index = feedback.index(feature)
            if prettify:
                yield((index, reference.pattern))
            else:
                yield((index, feature))


def ccr_top_compliments_concerns(df, prettify=True):
    """
    Get a list of the top compliments and concerns.

    Parameters
    ----------
        df (DataFrame) Augmented data.

    Returns
    -------
        Tuple of top compliments and top concerns
    """
    # Sort index to allow 1-to-1 comparison of topics in next block
    A = ccr_list_topics(df, 'Compliment').sort_index()
    B = ccr_list_topics(df, 'Concern').sort_index()

    # Filter topics by minimum score of 10% and where score in A or B is higher
    A_sub = A[A > B].sort_values(ascending=False).index.to_list()
    B_sub = B[B > A].sort_values(ascending=False).index.to_list()

    A_pret = ccr_prettify(A_sub, prettify=prettify)
    B_pret = ccr_prettify(B_sub, prettify=prettify)

    key = lambda x: x[0]

    A_iter = sorted(A_pret, key=key)
    B_iter = sorted(B_pret, key=key)

    compliments, concerns = [L for i, L in A_iter], [L for i, L in B_iter]
    return compliments[:3], concerns[:3]


def ccr_choose_top(dictionary, feedback_type, rank=1):
    """
    Get the nth (rank) top compliment or concern. Feeds into ccr_top_n.

    Parameters
    ----------
        df (dict): Top ranking compliments and concerns.
        
    Notes
    -----
        Assume input is a dictionary with keys representing feedback types
        and values representing lists.
    """
    if rank > len(dictionary[feedback_type]):
        return None
    return dictionary[feedback_type][int(rank) - 1]


def ccr_format(row):
    """ Format a single comment for insertion into a document. """
    base = config.format.feedback
    as_date = lambda dt: f"{dt: %m/%d/%Y}"
    
    message = base.format(Feedback_Comment=row.Feedback_Comment,
                          ResponseDateTime=as_date(row.ResponseDateTime),
                          RO=row.RO[6:])

    return ftfy.fix_text(message)


def ccr_choose_comments(df, feedback_type, n=20):
    """
    Return a list of comments, ROs, and ResponseDateTimes.

    Parameters
    ----------
        df (DataFrame): Augmented data.
        feedback_type (str): Compliment or Concern
        n (int): Number of comments to return
    
    Returns
    -------
        String
    
    Notes
    -----
        Comments are formatted for insertion into a document.
    """
    compliments, concerns = ccr_top_compliments_concerns(df, prettify=False)

    cond1 = (df[compliments] > 0).apply(any, axis=1)
    cond2 = (df[concerns] > 0).apply(any, axis=1)

    if feedback_type == 'Compliment':
        A, B, C = cond1, cond2, compliments[0]

    elif feedback_type == 'Concern':
        B, A, C = cond1, cond2, concerns[0]

    subset = df[A & ~B].copy()
    if subset.shape[0] == 0:
        subset = df[A].copy()

    values = [C, 'ResponseDateTime', 'tokens']
    subset.sort_values(values, ascending=False, inplace=True)

    comments = []
    iterator = iter(subset.sample(frac=.5).iterrows())
    
    while n > 0:
        try:
            comments.append(ccr_format(next(iterator)[1]))
        except StopIteration:
            break
        n -= 1
    return '\n\n'.join(comments)


def ccr_top(data):
    """ *REQUIRED* dummy method. Return output from prior method. """
    return data


def ccr_feedback(data):
    """ *REQUIRED* dummy method. Return output from prior method. """
    return data


def add_percent(number):
    """ Format a number into a percent. """
    return f"{number}%"


def remove_leading_zero(figure, regex=r"(?<=[^0-9.,])0(?=[1-9])|^ ?0(?=\d)"):
    """ Removes a redundant leading zero in a numeric or date figure. """
    re_ = re.compile(regex)
    remove_zero = lambda x: re_.sub(' ', str(x))
    remove_space = lambda x: str(x).strip().replace(" ", "")
    return remove_space(remove_zero(figure))


def format_number(number):
    """ Format integers to contain commas. """
    if pd.isnull(number):
        return number

    number = str(number)
    if number.replace(" ", "").isalpha():
        return number
    elif number.isdigit():
        return f"{int(number):,}"
    elif number.replace(".", "").isdigit():
        return f"{float(number):,.1f}"
    return number
