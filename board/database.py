# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:15:54 2019

database.py
@author: abastillasgl

Database Package (Fragment)
----------------

Classes to pull, push, and edit data in DBs. Requires the use of a config file
for connection parameters, queries, and other resources required to connect to
a database, run a query, and specify data needs.

This file is specific to the Board of Veterans Appeals and contains a short
list of the total available Table objects.

Dependencies:
    configuration (custom)
    pyodbc
    pandas
"""
from configuration import Configuration
from calculate import date_delta, normalize

from pyodbc import connect, OperationalError
from pandas import read_sql
import os

__config_file_1 = "./resources/configuration.yaml"

try:
    assert os.path.exists(__config_file_1)
except AssertionError:
    m = f"This package requires {__config_file_1}"
    raise AssertionError(m)

config = Configuration(__config_file_1)


class Table(object):
    """
    Base Class for other tables in VEODB.

    Table provides methods to query the database using connection information
    defined in __init__.
    
    Initialization
    
    Parameters
    ----------
        name (str): Name of the table in VEODB to query
        distinct (bool): Return distinct rows from queries. Default True.
        params (str): Connection parameters
        index_col (str): Name of column in Table to use as an index
    
    Attributes
    ----------
        name (str): Name of the table in VEODB to query
        distinct (bool): Return distinct rows from queries. Default True.
        params (str): Connection parameters
        index_col (str): Name of column in Table to use as an index
        columns (list): List of columns in this table
        size (int): Number of rows in this table
        shape (int): Number of rows and columns in this table
    
    Notes
    -----
        index_col is used to derive unique record counts for denominators
    """

    def __init__(self, name=None, distinct=True, params=None, index_col=None):
        self._name = name
        self._columns = None
        self._size = None
        self._shape = None
        self._distinct = distinct
        self._index_col = index_col
        self._params = params if params else config.params


    @property
    def name(self):
        return self._name


    @property
    def columns(self):
        if self._columns is not None:
            return self._columns

        try:
            query = config.select.ncol.format(table=self.name)
            columns = read_sql(query, self.connect()).columns
            self._columns = columns
            return self._columns
        except (TypeError, OperationalError):
            return None


    @property
    def size(self):
        if self._size:
            return self._size

        try:
            query = config.select.nrow.format(table=self.name)
            size = read_sql(query, self.connect()).values[0, 0]
            self._size = size
            return self._size
        except (TypeError, OperationalError):
            return None


    @property
    def shape(self):
        if not self._shape:
            nrow = self.size
            ncol = len(self.columns)

            if ncol is None:
                ncol = 0

            shape = (nrow, ncol)
            self._shape = shape

        return self._shape


    @property
    def params(self):
        return self._params


    @property
    def index_col(self):
        return self._index_col


    def __like(self, column, value):
        return config.where.like.format(**locals())


    def __notlike(self, column, string):
        return config.where.notlike.format(**locals())


    def __isnull(self, column):
        return config.where.isnull.format(**locals())


    def __notnull(self, column):
        return config.where.notnull.format(**locals())


    def __isin(self, column, values):
        return config.where.isin.format(**locals())


    def __notin(self, column, values):
        return config.where.notin.format(**locals())


    def __eq(self, column, value):
        return config.where.eq.format(**locals())


    def __noteq(self, column, value):
        return config.where.noteq.format(**locals())


    def __gt(self, column, value):
        return config.where.gt.format(**locals())


    def __lt(self, column, value):
        return config.where.lt.format(**locals())


    def __gte(self, column, value):
        return config.where.gte.format(**locals())


    def __lte(self, column, value):
        return config.where.lte.format(**locals())


    def connect(self, params=None):
        """
        Return a connection object

        Parameters
        ----------
            params (str): Connection parameters to use

        Returns
        -------
            Connection object
        """
        if params is None:
            params = self._params
        return connect(params)


    def where(self, **kwargs):
        """
        Return a string with conditions for a query

        Parameters
        ----------
            kwargs (dict): keywords for comparisons and values as lists of
                keywords and values according to the comparison type.

        Notes
        -----
            Comparison Type         Tuples
            ---------------         ------
            'like', 'notlike'       [(column1, value1), ...]
            'eq', 'noteq'           [(column1, value1), ...]
            'gt', 'gte'             [(column1, value1), ...]
            'lt', 'te'              [(column1, value1), ...]
            'isin', 'notin'         [(column1, (item1, item2, ...)), ...]
            'isnull', 'notnull'      (column1, column2, ...)

            Type                    Tuple
            ----                    ------
            type1                   [(column1, value1), ...]
            type2                   [(column1, (item1, item2, ...)), ...]
            type3                    (column1, column2, ...)

        Examples
        --------
            Tuple Type                              Example
            ----------                              -------
            [(column1, value1), ...]                [('Trust', 5), ...]
            [(column1, (item1, item2)), ...]        [('Trust', (4, 5)), ...]
             (column1, column2, ...)                 ('Comments', 'Age', ...)
        """
        type1 = [_ for _ in config.tuple.type1 if kwargs.get(_)]
        type2 = [_ for _ in config.tuple.type2 if kwargs.get(_)]
        type3 = [_ for _ in config.tuple.type3 if kwargs.get(_)]

        where = ''

        # Type 1: {column} STATEMENT {value}
        if type1:
            for _ in type1:
                conditional = getattr(config.where, _)
                keys_values = kwargs.get(_)

                conditionals = list()
                for key, value in keys_values:
                    v = conditional.format(column=key, value=value)
                    conditionals.append(v)
                conditionals.append(' ')
                where += ' AND '.join(conditionals)

        # Type 2: {column} IS (NOT) IN ({values})
        if type2:
            for _ in type2:
                conditional = getattr(config.where, _)
                keys_values = kwargs.get(_)

                conditionals = list()
                for key, value in keys_values:
                    value = """"'{"', '".join(value)}'"""
                    v = conditional.format(column=key, value=value)
                    conditionals.append(v)
                conditionals.append(' ')
                where += ' AND '.join(conditionals)

        # Type 3: {column} STATEMENT
        if type3:
            for _ in type3:
                conditional = getattr(config.where, _)
                columns = kwargs.get(_)

                conditionals = list()
                for column in columns:
                    v = conditional.format(column=column)
                    conditionals.append(v)
                where += ' AND '.join(conditionals)

        # Remove trailing AND
        where = where.strip()

        if where.endswith("AND"):
            where = where[:-3].strip()

        return where


    def run(self, query):
        """
        Execute a SQL query

        Parameters
        ----------
            query (str): SQL query to execute

        Notes
        -----
            Queries do not return anything.
        """
        connection = self.connect()
        connection.execute(query)
        connection.commit()
        connection.close()


    def select(self, *columns, n=10):
        """
        Randomly pull a specified number (n) of columns

        Parameters
        ----------
            columns (list): Columns and column adjacent commands to pass
            n (int): Number of rows to randomly select

        Notes
        -----
            Pass positional arguments for columns. Default number of rows to
            pull is 10.
        """
        query_args = dict(n=n, columns=', '.join(columns), table=self.name)
        query = config.select.sample.basic.format(**query_args)
        connection = self.connect()
        results = read_sql(query, connection)
        connection.close()
        return results


    def make_query(self, *args, **kwargs):
        """
        Return a query incorporating args and kwargs specified.

        Parameters
        ----------
            *args (list): List of columns to pull from DB
            **kwargs (dict): Where conditions for SQL query

        Returns
        -------
            Query as a string

        Notes
        -----
            Conditions keyword arguments specify the type of comparison to be
            made and the columns and values that need to be compared.

        Examples
        --------
            Comparison Type         Tuples
            ---------------         ------
            'like', 'notlike'       [(column1, value1), ...]
            'eq', 'noteq'           [(column1, value1), ...]
            'gt', 'gte'             [(column1, value1), ...]
            'lt', 'te'              [(column1, value1), ...]
            'isin', 'notin'         [(column1, (item1, item2, ...)), ...]
            'isnull', 'notnull'      (column1, column2, ...)

        Query Anatomy
        -------------
            SELECT {args; args can be aggregate funcs}
            FROM APG.{Table.name}
            WHERE {kwargs}

        """
        if not args:
            raise ValueError("No columns specified")

        # Query if this class has a table name specified. Otherwise raise error
        if self.name:

            # Basic query definitions
            table = self.name
            columns = ', '.join(args)
            conditions = None
            query = config.select.distinct

            # Add 'where' arguments
            if kwargs:
                conditions = self.where(**kwargs)
                query = config.select.where.distinct

            # Build query
            params = dict(table=table, columns=columns, conditions=conditions)
            return query.format(**params)

        raise AttributeError("No table specified")


    def pull(self, *args, **kwargs):
        """
        Return a DataFrame with data read in from a SQL query

        Parameters
        ----------
            *args (list): List of columns to pull from DB
            **kwargs (dict): `Where` conditions for SQL query

        Returns
        -------
            DataFrame

        Notes
        -----
            Conditions keyword arguments specify the type of comparison to be
            made and the columns and values that need to be compared.

        Examples
        --------
            Comparison Type         Tuples
            ---------------         ------
            'like', 'notlike'       [(column1, value1), ...]
            'eq', 'noteq'           [(column1, value1), ...]
            'gt', 'gte'             [(column1, value1), ...]
            'lt', 'te'              [(column1, value1), ...]
            'isin', 'notin'         [(column1, (item1, item2, ...)), ...]
            'isnull', 'notnull'      (column1, column2, ...)

        Query Anatomy
        -------------
            SELECT {args; args can be aggregate funcs}
            FROM APG.{Table.name}
            WHERE {kwargs}

        """
        # Query if this class has a table name specified. Otherwise raise error
        if self.name:
            query = self.make_query(*args, **kwargs)
            # Implement query
            print(f"RUNNING QUERY\n{query}")
            connection = self.connect()
            results = read_sql(query, connection, index_col=self._index_col)
            connection.close()
            return results
        raise AttributeError("No table specified")


class BoardDB(Table):
    """
    The BoardDB object interfaces with the VEO DB. BoardDB's methods allow end-
    users to query the Board DB and request specialized datasets for the creat-
    ion of the Board report.
    """

    def __init__(self, distinct=True, params=None, index_col='SurveyPersonID'):
        super().__init__(name=config.tables.board,
                         distinct=distinct,
                         params=params,
                         index_col=index_col)
        self.data = dict(tot=None, a11=None)


    def __getitem__(self, key):
        if key.lower() in set(['tot', 'trust over time']):
            key = 'tot'

        if key.lower() in set(['a11', 'a-11']):
            key = 'a11'

        return self.data.get(key)


    @property
    def a11(self):
        """ Return A-11 CX Domain Driver (a11) data """
        return self['a11']


    @property
    def tot(self):
        """ Return Trust over Time (TOT) data """
        return self['tot']


    def make_query_special(self, qtype, start=None, end=None, 
                           date_column='ResponseDateTime'):
        """
        Return an A11 or Trust over Time query

        Parameters
        ----------
            type (str): A-11 (a11) or Trust over Time (tot)
            start (str): Start date in ISO format
            end (str): End date in ISO format
            date_column (str): Name of table column with date to condition

        Returns
        -------
            DataFrame
        """
        query = config.select.tot
        if qtype.lower() == 'a11':
            query = config.select.a11

        if start is not None and end is not None:
            return (f"{query} WHERE {date_column} >= '{start}' "
                    f"AND {date_column} < '{end}'")
        elif start is not None:
            return f"{query} WHERE {date_column} >= '{start}'"

        elif end is not None:
            return f"{query} WHERE {date_column} < '{end}'"
            
        return query.format(date_column=date_column)


    def pull(self, start=None, end=None, diff=7, date_column='ResponseDateTime'):
        """
        Read a SQL query into an Trust over Time DataFrame.

        Parameters
        ----------
            start (str): Start date in ISO format
            end (str): End date in ISO format
            diff (int): Number of days difference for a datedelta if only `end`
            date_column (str): Name of table column with date to condition

        Returns
        -------
            DataFrame
        """
        columns = config.columns.board

        conditions = dict()
        if end:
            conditions['lt'] = [(date_column, f"'{end}'")]

        if start:
            conditions['gte'] = [(date_column, f"'{start}'")]

        if end is not None and start is None:
            delta = date_delta(end, days=diff)
            conditions['gte'] = [(date_column, f"'{delta[-1]}'")]
            conditions['lt'] = [(date_column, f"'{delta[0]}'")]

        query = self.make_query(*columns, **conditions)
        connection = self.connect()
        results = read_sql(query, connection)
        connection.close()
        return normalize(results)


    def pull_a11(self, start=None, end=None, date_column='ResponseDateTime'):
        """
        Read a SQL query into an A-11 Domain Drivers DataFrame.

        Parameters
        ----------
            start (str): Start date in ISO format
            end (str): End date in ISO format
            date_column (str): Name of table column with date to condition

        Returns
        -------
            DataFrame
        """
        query = self.make_query_special('a11', start, end, date_column)
        connection = self.connect()
        results_a11 = read_sql(query, connection)
        connection.close()

        self.data['a11'] = results_a11
        return results_a11


    def pull_tot(self, start=None, end=None, date_column='ResponseDateTime'):
        """
        Read a SQL query into an Trust over Time DataFrame.

        Parameters
        ----------
            start (str): Start date in ISO format
            end (str): End date in ISO format
            date_column (str): Name of table column with date to condition

        Returns
        -------
            DataFrame
        """
        query = self.make_query_special('tot', start, end, date_column)
        query = query.format(date_column=date_column)
        connection = self.connect()
        results_tot = read_sql(query, connection)
        connection.close()

        self.data['tot'] = results_tot
        return results_tot


if __name__ == "__main__":
    T = Table(config.tables.outpatient)
    b = BoardDB()
