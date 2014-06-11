from __future__ import division
import sys

#sys.path.append('/Users/oddrune/lib/NascenseAPI_v01e/')
#import emEvolvableMotherboard
#from ttypes import *

#from thrift import Thrift
#from thrift.transport import TSocket
#from thrift.transport import TTransport
#from thrift.protocol import TBinaryProtocol

from collections import namedtuple

import matplotlib.pyplot as plt

#Take a graph and a list of buffers, get how fit the sample buffer is.
#Each buffer represents 1 node.

#The color is decided my the overall voltage level in the buffer.
#The sorting is such that node 0 is buffer 0 in the list of buffers.
def fitness(graph, buffers):
    colors = []
    n = 0
    for buf in buffers:
        graph.nodes[n].color = sum(buf)/len(buf)
        n += 1
    
    return 0.0
