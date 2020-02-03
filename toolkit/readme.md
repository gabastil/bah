# Toolkit
Glenn Abastillas | November 10, 2019

This repository contains data structures, code, and resources that can be combined or reused in other packages.

Ideally, all unique scripts, data structures, and resources are kept here from other packages.

  1. [Git Location](#git_location)
  1. [Environments](#environments)
  1. [dependencies](#dependencies)
  1. [Inventory of Tools](#inventory)

#### Git Location <a id='git_location'></a>

The original repository is on Azure, but will be migrated to GitHub eventually.

https://dev.azure.com/abastillas/glenn/_git/toolkit

#### Environments <a id='environments'></a>

Toolkits were developed with a specific `conda` environment. This environment is stored in the `./resources/env.yaml` configuration file.

#### Dependencies <a id='dependencies'></a>

Dependencies and scripts that use them.

dependency Package | Used by | filename
:-- | :-- | :--
lxml | Configuration | configuration.py
xmljson | Configuration | configuration.py
yaml | Configuration | configuration.py
gensim | TopicModel | topic_model.py
svgwrite | Shape and its descendants | diagram.py

#### Inventory <a id='inventory'></a>

filename | Object(s) | Description
--: | :-- | :--
configuration.py | Configuration | Read and parse configuration files (e.g., yaml, json, xml)
database.py | Table | Objects to access databases
topic_model.py | TopicModel | Object that can read in text and create an LDA topic model
diagram.py | Shape, Region, DataSource, DataStorage, DataProcessing, Transport, UploadModule | Objects to read and draw diagrams for registry data flow operations
