import os, sys
import pathlib
import argparse
import numpy as np
import json
import hjson
import math
from termcolor import colored
import cv2 

# Import key paths depending on the machine running the code
# (must be harcoded in utils/default_param.json)
from utils.helper import read_json
import socket 
all_p = read_json(os.getcwd() +'/utils/default_param.json')
if socket.gethostname() in all_p[1]: 
    paths = all_p[1][socket.gethostname()]
    code_path, root_path, data_path = paths['code'], paths['root'], paths['data']
else: 
    code_path = str(pathlib.Path(__file__).parent.parent.absolute())
    root_path = str(pathlib.Path(__file__).parent.parent.parent.absolute())
    data_path = str(code_path) + '/data'


