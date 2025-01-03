#!/usr/bin/env python


import sys
import os

# add the custom module folder for inputs, with absoulte path with respect to script position
def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(get_script_path() + "/../../libs")
