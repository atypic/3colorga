from __future__ import division
import sys

import emEvolvableMotherboard
from ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import matplotlib.pyplot as plt
import random 

class Node:
    label = ""
    color = None
    neighbors = []
    visited = False
    def __init__(self, lab, col, nb, vis):
        self.label = lab
        self.color = col
        self.neighbors = nb
        self.visited = vis


class Monkey:
    """Thin wrapper for an monk... individual"""
    items = []
    recordings = []
    fitness = -1.0

    def __init__(self, sch):
        self.items = sch;

#Take a graph and a list of buffers, get how fit the sample buffer is.
#Each buffer represents 1 node.
#The color is decided my the overall voltage level in the buffer.
#The sorting is such that node 0 is buffer 0 in the list of buffers.
def fitness(graph, buffers):
    color(graph, buffers)
    return score(graph)
    
def score(graph):
    subSum = 0
    if graph.visited == True:
        return 0.0

    graph.visited = True
    if graph.neighbors != []:
        for n in graph.neighbors:
            subSum += distance(graph, n) + score(n)

    return subSum

def distance(a, b):
    return abs(a.color - b.color)

def color(node, buffers):
    """Traverse graph depth first, color it as the buffers suggest"""
    if node.color == None:
        node.color = sum(buffers[node.label])/len(buffers[node.label])
        for n in node.neighbors:
            color(n, buffers)
        return

def runMonkey(cli, monkey):
    """Runs population, returns a list of recordings (full objects)"""
    cli.reset()
    cli.clearSequences()
    recpins = []
    for w in monkey.items:
        if w.operationType == emSequenceOperationType().RECORD:
            for i in w.pin:
                recpins.append(i)
        cli.appendSequenceAction(w)

    print "Running sequences"
    cli.runSequences()
    cli.joinSequences()
    
    for r in recpins:
        monkey.recordings.append(cli.getRecording(r))

    return monkey

def initPop(size, allowed_types):
    """Initializes a list of Monkey objects, len == size; each Monkey based on allowed_types."""
    population = []
    for ind in xrange(0,size):
        individual = []
        #limits.
        sequenceRunTimeMs=200
        maxPins = 16
        unusedPins = set(range(0,maxPins))
        numNodes = 4

        usedDACchannels = 0
        recordPins = []
        #First the recording pins for this individual
        for j in xrange(0,numNodes):
            foo = random.choice(list(unusedPins))
            recordPins.append(foo)
            unusedPins.remove(foo)

        for r in recordPins:
            it = emSequenceItem()
            it.pin = [r]
            it.startTime = 0
            it.endTime = sequenceRunTimeMs
            it.frequency = 1000
            it.operationType = emSequenceOperationType().RECORD   #implies analogue 
            individual.append(it)

        #Use remaining pins for DA channels.
        for i in range(0, 8):
            it = emSequenceItem()
            pin = random.sample(unusedPins, 1)
            unusedPins.remove(pin[0])
            
            it.pin = pin
            it.startTime = 0
            it.endTime = sequenceRunTimeMs
            it.amplitude = random.randrange(0,255)
            it.operationType = emSequenceOperationType().CONSTANT   #implies analogue 
            individual.append(it)

        population.append(Monkey(individual))

    return population

def select(pop):
    return pop

def mutatePopulation(pop):
    return pop

def gaLoop(cli, generations):
    #create cli object here
    nodes = []
    for i in range(0,4):
        nodes.append(Node(i, None, [], False))

    graph = nodes[0]
    nodes[0].neighbors = [nodes[1],nodes[2]]
    nodes[1].neighbors = [nodes[0],nodes[2],nodes[3]]
    nodes[2].neighbors = [nodes[0],nodes[1]]
    nodes[3].neighbors = [nodes[1]]

    population = initPop(20, "")
    for generation in xrange(0,generations):
        for monkey in population:
            print "Running monkey"
            monkey.fitness = fitness(graph, runMonkey(cli, monkey))
        fitMonkeys = select(population)
        print "Most fit monkey of generation ", generation, " :", fitMonkeys[0]

        population = mutatePopulation(fitMonkeys)
