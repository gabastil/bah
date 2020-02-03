# -*- coding: utf-8 -*-
"""
configuration.py
Glenn Abastillas
Created on Thu Oct 31 09:48:49 2019

Generic Configuration Class for configuration files in YAML, XML, or JSON.

In addition to the Config class being able to parse YAML, XML, JSON formats,
this class can also convert the specified file to any of those three formats.

"""
from collections import namedtuple
from pathlib import Path
from xmljson import BadgerFish as bfl
from datetime import datetime
import re
import os
import yaml
import json
import lxml


class Configuration(object):
    """
    This class loads and parses configuration files in YAML, XML, and JSON
    formats to allow for dot-notation access to the parameters defined in those
    files. Additionally, this class can convert from any of those three formats
    into any of the others (i.e., from YAML/XML/JSON to YAML/XML/JSON formats).

    Example usage:

        `configuration.xml`

        >>> <configuration>
        >>>     <parameters>
        >>>         <parameter1>
        >>>             value1
        >>>         </parameter1>
        >>>         <parameter2>
        >>>             value2
        >>>         </parameter2>
        >>>     </parameters>
        >>> </configuration>

        `configuration.py`

        >>> config = Configuration('configuration.xml')
        >>> config.parameters.parameter1
        >>> "value1"
        >>>
        >>> config.parameters.parameter2
        >>> "value2"
        >>>
    """
    EXT = ['yml', 'yaml', 'json', 'xml', 'html', 'htm']
    TYPE = ['yaml'] * 2 + ['json'] * 1 + ['xml'] * 3
    PARSER = [yaml.load] * 2 + [json.load] * 1 + [lxml.etree.parse] * 3
    PATH_FINDER = re.compile(r"(~|(\w:|\.)[/\\])", re.I)

    def __init__(self, fn="./resources/configuration.yaml"):
        """
        Initialize Config object. This class will look in a resources folder by
        default.

        Parameters
        ----------
            fn (str): Path to configuration file.

        Notes
        -----
            __init__() will try to open ../resources/configuration.yaml by
            default. Otherwise, nothing will happen and a fill will need to be
            loaded.
        """

        self._fn = Path(fn)
        self._parser = self.__get_parser(self._fn)
        self._type = self._filetype(fn)
        self._today = datetime.today()
        try:
            self.__initialize()
        except IOError:
            raise Warning("File not found")
            

    @property
    def today(self):
        return self._today

    @property
    def fn(self):
        return self._fn
    

    @fn.setter
    def fn(self, fn):
        self._fn = fn
        

    @property
    def parser(self):
        return self._parser
    

    def __initialize(self):
        """ Load and assign attributes from configuration file to object """
        configuration = self.__load_file()

        for k, v in configuration.items():
            attribute = self.__add_option(k, v)
            setattr(self, k, attribute)
            

    def __add_option(self, key, value):
        """ Build a namedtuple data structure """
        if not isinstance(value, dict):
            if isinstance(value, list):
                dict_found = [isinstance(v, dict) for v in value]

                if not any(dict_found):
                    return value

                else:
                    values_ = []
                    for value_ in value:
                        if isinstance(value_, dict):
                            for k, v in value_.items():
                                aor = self.__add_option(k, v)
                                values_.append(aor)
                    return values_
            else:
                is_filepath = self.PATH_FINDER.match(str(value))

                if is_filepath:
                    return Path(value).expanduser()

                return value

        # One of the values in value had a dict
        values_ = []

        for k, v in value.items():
            values_.append(self.__add_option(k, v))

        container = namedtuple(key, value.keys())
        return container(*values_)
    

    def __get_parser(self, fn):
        """ Return a yaml, json, or xml parser """
        F = self._filetype(fn)
        i = self.EXT.index(F)
        self.type = F
        return self.PARSER[i]
    

    def __load_file(self):
        """ Load the configuration file from this class's self.fn attribute """
        with open(self._fn, 'r') as cin:
            if self._type == 'xml':
                C = cin.read()
                X = lxml.etree.fromstring(C)
                configuration = self.__xml_to_json(X)
            elif self._type in ('yaml', 'yml'):
                configuration = self._parser(cin, yaml.Loader)
            else:
                configuration = self._parser(cin)
        return configuration
    

    def __xml_to_json(self, xml):
        """ Convert an ElementTree into a dictionary """
        children = xml.getchildren()

        if not children:
            return xml.text.strip()

        children_ = []

        for child in children:
            child_ = self.__xml_to_json(child)
            children_.append(child_)

        key = xml.tag
        value = children_ if len(children_) > 1 else children_[0]

        return {key: value}
    

    def _filetype(self, fn):
        """ Determine the format of an input file path via its extension """
        _, extension = os.path.basename(fn).split(".")

        assert extension.lower() in self.EXT

        for i, ext in enumerate(self.EXT):
            if extension.lower() == ext:
                self._type = self.TYPE[i]
                return ext
            

    def to_json(self, fn=None, fp=None):
        """
        Convert the configuration file this class was instantiated with to JSON

        Parameters
        ----------
            fp (str): Path to save JSON file. Default is current directory.
            fn (str): Filename to use. Default is the filename in this class.
        """
        configuration = self.__load_file()

        if not fn:
            fn, extension = os.path.basename(self._fn).split(".")

        output_fp = f"{fn}.json"

        if fp:
            output_fp = Path(fp) / output_fp

        with open(output_fp, "w") as cout:
            json.dump(configuration, cout)
            

    def to_xml(self, fn=None, fp=None):
        """
        Convert the configuration file this class was instantiated with to XML

        Parameters
        ----------
            fp (str): Path to save XML file. Default is current directory.
            fn (str): Filename to use. Default is the filename in this class.
        """
        configuration = self.__load_file()

        if not isinstance(configuration, dict):
            configuration = {'configuration': configuration}

        configuration = bfl().etree(configuration)[0]
        element_tree = lxml.etree.ElementTree(configuration)

        if not fn:
            fn, extension = os.path.basename(self._fn).split(".")

        output_fp = f"{fn}.xml"
        if fp:
            output_fp = Path(fp) / output_fp

        element_tree.write(output_fp, pretty_print=True)
        

    def to_yaml(self, fn=None, fp=None):
        """
        Convert the configuration file this class was instantiated with to YAML

        Parameters
        ----------
            fp (str): Path to save YAML file. Default is current directory.
            fn (str): Filename to use. Default is the filename in this class.
        """
        configuration = self.__load_file()

        if not fn:
            fn, extension = os.path.basename(self._fn).split(".")

        output_fp = f"{fn}.yaml"
        if fp:
            output_fp = Path(fp) / output_fp

        with open(output_fp, "w") as cout:
            yaml.dump(configuration, cout)

if __name__ == "__main__":
    # Create folder structure
    c = Configuration()
    
    folders = [c.paths.board, 
               c.paths.data, 
               c.paths.images,
               c.paths.midmonth, 
               c.paths.month,
               c.paths.output,
               c.paths.resources, 
               c.paths.template, 
               c.paths.week,]
    
    for folder in folders:
        folder.mkdir(exist_ok=True)
