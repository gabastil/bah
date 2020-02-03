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
import os
import yaml
import json
import lxml


class Configuration():
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
    EXT = ['yaml', 'json', 'xml', 'html', 'htm']

    def __init__(self, fn="../resources/configuration.yaml"):
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
        try:
            self.__initialize()
        except IOError:
            pass

    @property
    def fn(self):
        """ Get the filename this class reads from """
        return self._fn

    @fn.setter
    def fn(self, fn):
        """  Set the filename this class reads from """
        self._fn = fn

    @property
    def parser(self):
        """ Return the parser type based on the filename """
        return self._parser

    def __add_option_recursion(self, key, value):
        """ Build a namedtuple data structure """
        if not (isinstance(value, dict) or isinstance(value, list)):
            return value
        
        values_ = []
        
        if isinstance(value, dict):
            container = namedtuple(key, value.keys())
            dict_found = [isinstance(v, dict) for k, v in value.items()]
    
            if not any(dict_found):
                return container(*value.values())
    
            for k, v in value.items():
                assignment = self.__add_option_recursion(k, v)
                values_.append(assignment)

            return container(*values_)

        for item in value:
            if isinstance(item, dict):
                k_v = list(item.items())
                k, v = k_v[0]                
                item = self.__add_option_recursion(k, v)
            values_.append(item)
        return values_

    def __filetype(self, fn):
        """ Determine the format of an input file via its extension """
        _, extension = os.path.basename(fn).split(".")

        assert extension.lower() in self.EXT

        filetype = None

        if extension.lower() == self.EXT[0]:
            filetype = self.EXT[0]

        if extension.lower() == self.EXT[1]:
            filetype = self.EXT[1]

        if extension.lower() in self.EXT[2:]:
            filetype = self.EXT[2]

        return filetype

    def __get_parser(self, fn):
        """ Return a yaml, json, or xml parser """
        filetype = self.__filetype(fn)
        parser = None

        if filetype == self.EXT[0]:
            parser = yaml.load

        if filetype == self.EXT[1]:
            parser = json.load

        if filetype == self.EXT[2]:
            parser = lxml.etree.parse

        return parser

    def __initialize(self):
        """ Load and assign attributes from configuration file to object """
        configuration = self.__load_file()

        for k, v in configuration.items():
            attribute = self.__add_option_recursion(k, v)
            setattr(self, k, attribute)

    def __load_file(self):
        """ Load the configuration file from this class's self.fn attribute """
        with open(self._fn, 'r') as cin:
            if self.__filetype(self._fn) == 'xml':
                cin = cin.read()
                configuration = self.__xml_to_json(lxml.etree.fromstring(cin))
            elif self.__filetype(self._fn) == 'yaml':
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
    c = Configuration()