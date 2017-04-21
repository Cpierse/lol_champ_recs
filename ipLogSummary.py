# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:22:26 2017
Reads an ip address log file and spits out a list of all ip addresses and 
the number of requests. Run from command line:

python ipLogSummary.py logfile.txt 

@author: Chris Pierse
"""
from collections import Counter
import sys

# Read the log file
filename = sys.argv[1]
# IPs are the first element on each line
ips = [line.split(' ')[0] for line in open(str(filename))]
counts = Counter(ips)
print(counts)
