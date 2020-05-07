import os, sys
import argparse
import numpy as np
import json
import hjson
import math
from termcolor import colored
import cv2 

# Import key paths depending on the machine running the code
from utils.helper import read_json
import socket 
all_p = read_json(os.getcwd() +'/utils/default_param.json')
paths = all_p[1][socket.gethostname()]

code_path, root_path, data_path = os.getcwd(), os.getcwd(), os.getcwd()
