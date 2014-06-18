from __future__ import division
import sys

import emEvolvableMotherboard
from ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

#import matplotlib.pyplot as plt
#mpl.use('pdf')
import random 
import copy

import numpy as np
import scipy.stats as st

class Node:
    label = ""
    color = None
    wantedColor = None
    neighbors = []
    visited = False
    def __init__(self, lab, col, nb):
        self.label = lab
        self.wantedColor = col
        self.neighbors = nb


class Monkey:
    """Thin wrapper for a monkey... individual"""
    items = list()
    recordings = list()
    fitness = -1.0
    graph = list()

    def __init__(self, sch):
        self.items = sch;
        self.recordings = list()
        self.fitness = -1.0
        self.graph = list()

#Take a graph and a list of buffers, get how fit the sample buffer is.
#Each buffer represents 1 node.
#The color is decided my the overall voltage level in the buffer.
#The sorting is such that node 0 is buffer 0 in the list of buffers.
def fitness(graph, monkey):
    return score(graph)

def score(node):
    """Score the subgraph from node."""
    subSum = 0
    if node.visited == True:
        return 0.0

    node.visited = True
    if node.neighbors != []:
        for n in node.neighbors:
#            subSum += distance(node, n) + score(n)
            subSum += abs(node.wantedColor - node.color) + score(n)

    return subSum

def distance(a, b):
    return abs(a.color - b.color)

def color(node, buffers):
    """Traverse graph depth first, color it as the buffers suggest"""
    if node.color == None:
        if(len(buffers[node.label].Samples) != 0):
            node.color = sum(samplesToVolts(buffers[node.label].Samples))/len(buffers[node.label].Samples)
        else:
            node.color = -1.0
        for n in node.neighbors:
            color(n, buffers)

def printColors(node, visitedList):
    if  node.label in visitedList:
        return;
    else:
        print node.label," : ", node.color, " : ", node.wantedColor
        visitedList.append(node.label)
        for c in node.neighbors:
            printColors(c, visitedList)


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

    cli.runSequences()
    cli.joinSequences()
   
    for r in recpins:
        monkey.recordings.append(copy.deepcopy(cli.getRecording(r)))

    return monkey

def initPop(size, allowed_types):
    """Initializes a list of Monkey objects, len == size; each Monkey based on allowed_types."""
    population = []
    for ind in xrange(0,size):
        individual = []
        #limits.
        sequenceRunTimeMs=500
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
            it.frequency = 10000
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
    """Select uses tournament selection and returns ALL NEW objects. Population is a list of Monkeys"""

    sortedByFitness = sorted(pop, key=lambda m: m.fitness, reverse=False)

    return sortedByFitness[0:10]

#Breed a new population based on passed population
#Also mutates
def breed(size, pop):
    bredPopulation = []
    for i in xrange(0,size):
        parent = random.choice(pop)
        child = Monkey(copy.deepcopy(parent.items))
        #Mutate one pin
        mutateItem = random.randint(0,11)
        if child.items[mutateItem].operationType == emSequenceOperationType().CONSTANT:
            child.items[mutateItem].amplitude += max(255,random.gauss(1,1))
        bredPopulation.append(child)

    return bredPopulation

def samplesToVolts(buf):
    return [i * (5.0/4096.0) for i in buf]

def gaLoop(cli, generations):
    #create cli object here
    nodes = []
    for i in range(0,4):
        nodes.append(Node(i, None, []))

    """
    Wanted graph:
     A \
    |   C --- A 
     B /

    Well, that's the coloring anyway.
    """
    #TODO: Generate this.
    graph = nodes[0]
    nodes[0].neighbors = [nodes[1],nodes[2]]
    nodes[0].wantedColor = 5.0
    nodes[1].neighbors = [nodes[0],nodes[2],nodes[3]]
    nodes[1].wantedColor = -5.0
    nodes[2].neighbors = [nodes[0],nodes[1]]
    nodes[2].wantedColor = 0.0
    nodes[3].neighbors = [nodes[1]]
    nodes[3].wantedColor = 5.0

    numIndividuals = 50
    population = initPop(numIndividuals, "")
    for generation in xrange(0,generations):
        print "Running generation ", generation
        for monkey in population:
            monkeyRun = runMonkey(cli, monkey)
            coloredGraph = copy.deepcopy(graph)
            color(coloredGraph, monkeyRun.recordings)

            monkey.graph = coloredGraph
            monkey.fitness = fitness(coloredGraph, monkey)

        fitMonkeys = select(population)
        print "Most fit monkey of generation ", generation, " : ", fitMonkeys[0].fitness
        printColors(fitMonkeys[0].graph, list())
        #This should return a brand new set of monkeys
        population = breed(numIndividuals, fitMonkeys)
