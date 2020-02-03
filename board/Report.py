#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:21:10 2018

@author: glenabastillas

REPORT PACKAGE
-------------------------------------------------------------------------------
Glenn Abastillas | November 7, 2018 | Report.py
-------------------------------------------------------------------------------

These methods are used to build a document report using the Variable Replace-
ment Framework (VRF). This framework aims to minimize the need to modify the
script to change parameters. To aid in this goal, this framework relies on add-
itional resources in folders (e.g., ./data, ./template) that are user-defined.

The Classes defined in this package rely on three types of resources:
        data: Configuration files, mappings, data from other sources
        
        template: Pre-formatted documents (e.g., Word, PowerPoint) with var-
            iable names embedded throughout. 
            
            These names follow the format: $NAME (all-caps)
            
            They are placed throughout the document where the variable would
            normally appear.

Dependencies:
    pandas
    docx
    calculate (custom package)
    database_access (custom package)
"""
from writer import query, analyze, build_tables, compose
from writer import document, variables, config
from pathlib import Path
from calculate import date_month, date_format
import time
import pickle

this = config.script.report


class Report(object):
    """
    Base class for all Report objects.
    """
    
    def __init__(self, template, variables, vsignals_data, fn, fp):
        self.template = template
        self.variables = variables
        self.vsignals_data = vsignals_data
        self.fn = fn
        self.fp = fp

    def build(self):
        """ Compose a document using external data (e.g., SQL results) """
        raise NotImplementedError()

    def save(self, fn):
        """ Save current document as specified name """
        raise NotImplementedError()

class BoardReport(Report):
    """ 
    The BoardReport Class and API streamline the report creation process
    for the weekly, mid-monthly, and monthly analytics report for the
    Board of Veterans' Appeals (The Board). This class is uses an implemen-
    tation of the Variable-Replacement-Framework (VRF).
    
    --- Variable-Replacement-Framework (VRF) ---
    The VRF framework focuses on the analytics portion of an analytics
    report rather than formatting automation upon report generation.
    
    The VRF paradigm requires two files to function. The first is the out-
    put report template (e.g., Word document). This report template should 
    already be formatted and contain replaceable variable placeholders.
    The second is the instructional spreadsheet mapping. This mapping cont-
    ains a list of variables, data source, analytic method, and other cond-
    itions required to perform the analysis. The variable names in this
    document need to match the variable names in the templated document.
    
    An example of how VRF works is as follows:
        
        [Report Tempalte]
        "... for the $N respondents, they showed $P Trust ..."
        
        [Post-VRF Processing]
        "... for the 1,234 respondents, they showed 50% Trust ..."
        
        [Mapping]
        $VAR   Description               Analytic Method
        $N     Number of Respondents     number_of_respondents()
        $P     Trust Percent             calculate_trust_by()
    """
    
    def __init__(self, template=config.paths.document, 
                       variables=config.paths.variables,
                       vsignals_data=None,
                       fn="The_Board_Report",
                       fp=config.paths.output):

        args = {"template": Path(template),
                "variables": Path(variables),
                "fn": Path(fn),
                "fp": Path(fp),
                "vsignals_data": vsignals_data}

        super(BoardReport, self).__init__(**args)

    def build(self, 
              delta="week", 
              end=config.today, 
              save=False, 
              verbose=True,
              use_existing=None,
              fn=None,
              method='r'):
        """ 
        Create a Board report for a specified time period. Return the three
        datasets used to build the report as a dictionary with each period
        name as the key and the dataset as a value.
        
        Parameters
        ----------
            delta (str): Time period as 'week', 'mid-month', or 'month'
            end (str, Date): What day the report should end and exclude
            save (bool): Save the pulled data set?
            verbose (bool): Print out each major step in the build process?
            use_existing (bool): Use existing data in the ./data folder?
            fn (str): File name for query results
            method (str): Type of A11 CX Domain calculation to use
        
        Returns
        -------
            Dict with the following structure {'df': pd.DataFrame, ...}
        
        Notes
        -----
            2019-12-16: Updated default method to be by response `r`    
        
            A11 CX Domain calculations are either by Question "Q"/"q" or by
            Response "R"/"r". Default is by question.
            
            
        """
        start_ = time.time()
        if use_existing:
            filename = config.paths.data / f"{use_existing}.pkl"

            if verbose:
                print(f">> Reading in existing data {filename}")
            
            if filename.exists():
                with open(filename, 'rb') as existing_data:
                    data = pickle.load(existing_data)
            else:
                error_message = f"File not found in {config.paths.data}"
                raise FileNotFoundError(error_message)
        else:
            if verbose:
                print(">> Pulling data")
            data = query(delta, end, method=method, fn=fn, save=save)

            if verbose:
                print(">> Partitioning data")
                
            if ('mid' in delta.lower()) and (data['dp'].shape[0] == 0):
                date_tuple = (config.today.year, config.today.month, 1)
                endm, startm, *__ = date_month(date_tuple)
                
                condition1 = data['d90'].ResponseDateTime >= startm
                condition2 = data['d90'].ResponseDateTime < endm
    
                data['dp'] = data['d90'][condition1 & condition2]
            
        if(verbose):
            print(">> Analyzing data\n")
        analyze(variables, data, a11=method)
        tables = build_tables(data)
        compose(document, variables, tables)

        try:
            date_ = f"{date_format(end, asdatetime=1):%Y%m%d}"
            args_ = dict(fn=self.fn, delta=delta.title(), date=date_)

            filename = config.filenames.board.format(**args_)
            
            parent, child = Path(self.fp), Path(self.fp / delta)
            parent.mkdir(exist_ok=1)
            child.mkdir(exist_ok=1)
#            output_path = self.fp / delta / filename
            document.save(child / filename)
            message = "Board report completed in"
        except PermissionError:
            message = "(!) ERROR: File open, did not save.\n\nCompleted in"

        # Information to be printed to console
        finish_ = time.time()
        run_time= finish_ - start_

        if run_time > 60:
            run_time = f"{round(run_time / 60)} minutes"
        else:
            run_time = f"{round(run_time, 1)} seconds"
            
        content = dict(fp_label='File saved'.ljust(10),
                       fp=self.fp / delta, 
                       fn_label='File name'.ljust(10), 
                       fn=filename,
                       delta=delta, 
                       end=end,
                       d90='90-Day Period'.rjust(20), 
                       d90_shape=data['d90'].shape,
                       dn='Current Period'.rjust(20), 
                       dn_shape=data['dn'].shape,
                       dp='Previous Period'.rjust(20), 
                       dp_shape=data['dp'].shape,
                       message=message, 
                       run_time=run_time)

        print(this.build.infostring.format(**content))
        return(data)
    
    def build_weekly_report(self, *args, **kwargs):
        return self.build('week', *args, **kwargs)

    def build_midmonthly_report(self, *args, **kwargs):
        return self.build('midmonth', *args, **kwargs)
    
    def build_monthly_report(self, *args, **kwargs):
        return self.build('month', *args, **kwargs)

if __name__ == "__main__":
    """ Main executable part of the Board script. """    
    from optparse import OptionParser
    
    parser = OptionParser()
    
    help_ = this.main.optparse.help
    flags = this.main.optparse.flags
    today = config.today   
    
    kwargs= [dict(dest='period', type=str, help=help_.time, default='week'),
             dict(dest='end_date', type=str, help=help_.date, default=today),
             dict(dest='save', action='store_true', default=False),
             dict(dest='use_existing', type=str, help=help_.file),
             dict(dest='fn', type=str, help=help_.save),
             dict(dest='method', type=str, help=help_.a11, default='r'),]
    
    for a_, k_ in zip(flags, kwargs):
        parser.add_option(*a_, **k_)

    (options, args) = parser.parse_args()
    
    # Determine the function to use based on the reporting period specified
    if options.period.startswith('month'):
        func = BoardReport().build_monthly_report
        
        # Set the end date to be the first of the current month
        year, month, day = date_format(options.end_date, astuple=True)
        options.end_date = date_format((year, month, 1))
        
    elif options.period.startswith('mid'):
        func = BoardReport().build_midmonthly_report    
        
    elif options.period.startswith('week'):    
        func = BoardReport().build_weekly_report

    else:
        error_message = this.main.error
        raise AttributeError(error_message)
        
    a11_response_type = 'response' if options.method in 'Rr' else 'question'
    
    save_query_result = 'saved' if options.save else 'discarded'
    if options.use_existing:
        save_query_result = 'loaded from previous data'
    
    content = [options.period.title(),
               date_format(options.end_date),
               a11_response_type,
               save_query_result,
               config.paths.board,
               options.period,
               ]
    
    print(this.main.infostring.format(*content))
    
#    data = func(end=options.end_date, 
#                fn=options.fn, 
#                save=options.save, 
#                use_existing=options.use_existing,
#                method=options.method)
