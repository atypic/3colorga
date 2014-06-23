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
import itertools

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


class Item:
    item = None
    inputPin = -1
    def __init__(self, it, ip):
      self.item = it;
      self.inputPin = ip;

class Monkey:
    """Thin wrapper for a monkey... individual"""
    items = list()
    recordings = list()
    fitness = -1.0
    graphs = list()

    def __init__(self, sch):
        self.items = sch;
        self.recordings = list()
        self.fitness = -1.0
        self.graphs = list()

#Take a graph and a list of buffers, get how fit the sample buffer is.
#Each buffer represents 1 node.
#The color is decided my the overall voltage level in the buffer.
#The sorting is such that node 0 is buffer 0 in the list of buffers.
def fitness(monkey):
    s = 0
    for g in monkey.graphs:
        s += score(g)
    return s

def score(node):
    """Score the subgraph from node."""
    subSum = 0.0
    if node.visited == True:
        return 0.0

    node.visited = True
    if node.neighbors != []:
        for n in node.neighbors:
            subSum += score(n)

    return abs(node.wantedColor - node.color) + subSum

def distance(a, b):
    return abs(a.color - b.color)

def color(node, buffers):
    """Traverse graph depth first, color it as the buffers suggest"""
    if node.color == None:
        if(len(buffers[node.label].Samples) != 0):
            node.color = sum(samplesToVolts(buffers[node.label].Samples))/len(buffers[node.label].Samples)
            #print "Colored node ", node.label, " ", node.color, "buffsize", len(buffers[node.label].Samples)
        else:
            node.color = None
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


def printMonkey(monkey):
  print "------------------"
  print "------------------"

def runMonkey(cli, monkey, inputValue):
    """Runs population, returns a list of recordings (full objects)"""

    cli.reset()
    cli.clearSequences()
    recpins = []
    for w in monkey.items:
        """This is a input pin, digital one."""
        if w.inputPin != -1:
            w.item.frequency = inputValue[w.inputPin]
            w.item.cycleTime = 100

        elif w.item.operationType == emSequenceOperationType().RECORD:
            for i in w.item.pin:
                recpins.append(i)

        #print w.item
        cli.appendSequenceAction(w.item)
    #set up the input pins.
    cli.runSequences()
    cli.joinSequences()
   
    for r in recpins:
        monkey.recordings.append(copy.deepcopy(cli.getRecording(r)))

    return monkey

def initPop(size, allowed_types):
    """Initializes a list of Monkey objects, len == size; each Monkey based on allowed_types."""
    population = []
    for ind in xrange(0,size):
        #print " -New individual - "
        individual = []
        #limits.
        sequenceRunTimeMs=100
        maxPins = 16
        unusedPins = set(range(0,maxPins))
        numNodes = 4
        numInputPins = 2

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
            individual.append(Item(it, -1))

        for i in xrange(0, numInputPins):
            it = emSequenceItem()
            foo = random.choice(list(unusedPins))
            unusedPins.remove(foo)
            it.pin = [foo]
            it.startTime = 0
            it.endTime = sequenceRunTimeMs
            it.operationType = emSequenceOperationType().DIGITAL
            individual.append(Item(it, i))

        #Use remaining pins for DA channels.
        for d in range(0, 8):
            it = emSequenceItem()

            pin = random.sample(unusedPins, 1)
            unusedPins.remove(pin[0])
            
            it.pin = pin
            it.startTime = 0
            it.endTime = sequenceRunTimeMs
            #it.amplitude = random.randrange(0,255)
            it.amplitude = 255
            it.operationType = emSequenceOperationType().CONSTANT   #implies analogue 
            individual.append(Item(it, -1))

        newMonkey = Monkey(individual)

        population.append(newMonkey)

    return population

def select(pop):
    """Select uses tournament selection and returns ALL NEW objects. Population is a list of Monkeys"""
    """len(pop)/2 tournaments are held."""

    winners = []
    for t in xrange(0,int(len(pop)/2)):
      fighters = random.sample(pop, 2)
      sortedByFitness = sorted(fighters, key=lambda m: m.fitness, reverse=False)
      if random.random() < 0.9:
        winners.append(sortedByFitness[0])
      else:
        winners.append(sortedByFitness[1])
    
    return winners

#Breed a new population based on passed population
#Also mutates
def breed(size, pop):
  bredPopulation = []
  while len(bredPopulation) < size:

    mom = random.choice(pop)
    mom2 = random.choice(pop)

    child = Monkey(copy.deepcopy(mom.items))
    child2 = Monkey(copy.deepcopy(mom2.items))

    j = 0
    xpoint = random.randrange(0,16)
    for i,i2 in zip(child.items, child2.items):
      it = i.item
      it2 = i2.item

      if j > xpoint:
        it.pin = it2.pin
      else:
        it2.pin = it.pin

      j += 1

      #First mutation
      if it.operationType == emSequenceOperationType().CONSTANT:
        if random.random() > 0.96:
          it.amplitude += int(min(255,random.gauss(2,2)))

      if it2.operationType == emSequenceOperationType().CONSTANT:
        if random.random() > 0.96:
          it2.amplitude += int(min(255,random.gauss(2,2)))


    #Second mutation operator ... kinda like cross over tbh.
    #if random.random() > 0.99:
    #  swapItems = random.sample(range(0,12), 2)
    #  tmp = child.items[swapItems[0]].pin
    #  child.items[swapItems[0]].pin = child.items[swapItems[1]].pin
    #  child.items[swapItems[1]].pin = tmp

    #if random.random() > 0.99:
    #  swapItems = random.sample(range(0,12), 2)
    #  tmp = child2.items[swapItems[0]].pin
    #  child2.items[swapItems[0]].pin = child2.items[swapItems[1]].pin
    #  child2.items[swapItems[1]].pin = tmp




    if isValidMonkey(child):
      bredPopulation.append(child)
    if isValidMonkey(child2):
      bredPopulation.append(child2)

  return bredPopulation

def isValidMonkey(monkey):
  uniqueRecPins = []
  for it in monkey.items:
    if it.item.operationType == emSequenceOperationType().RECORD:
      if it.item.pin in uniqueRecPins:
        return False
      else:
        uniqueRecPins.append(it.item.pin)

  if len(uniqueRecPins) != 4:
    return False

  return True

def samplesToVolts(buf):
    return [i * (5.0/4096.0) for i in buf]

def allPossibleColorings(graph):
  R = 5.0
  G = -5.0
  B = 0.0

  ret = []
  g = copy.deepcopy(graph)
  g[0].wantedColor = B
  g[1].wantedColor = G
  g[2].wantedColor = R
  g[3].wantedColor = R
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = B
  g[1].wantedColor = R
  g[2].wantedColor = G
  g[3].wantedColor = G
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = G
  g[1].wantedColor = B
  g[2].wantedColor = R
  g[3].wantedColor = R
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = B
  g[1].wantedColor = G
  g[2].wantedColor = B
  g[3].wantedColor = R
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = B
  g[1].wantedColor = R
  g[2].wantedColor = B
  g[3].wantedColor = G
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = G
  g[1].wantedColor = B
  g[2].wantedColor = G
  g[3].wantedColor = R
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = R
  g[1].wantedColor = G
  g[2].wantedColor = R
  g[3].wantedColor = B
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = G
  g[1].wantedColor = R
  g[2].wantedColor = G
  g[3].wantedColor = B
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = R
  g[1].wantedColor = B
  g[2].wantedColor = R
  g[3].wantedColor = G
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = R
  g[1].wantedColor = G
  g[2].wantedColor = B
  g[3].wantedColor = B
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = G
  g[1].wantedColor = R
  g[2].wantedColor = B
  g[3].wantedColor = B
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = R
  g[1].wantedColor = B
  g[2].wantedColor = G
  g[3].wantedColor = G
  ret.append(g)

  return ret

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
    graph = nodes[0]
    nodes[0].neighbors = [nodes[1],nodes[2]]
    nodes[1].neighbors = [nodes[0],nodes[2],nodes[3]]
    nodes[2].neighbors = [nodes[0],nodes[1]]
    nodes[3].neighbors = [nodes[1]]
    
    #nodes[0].wantedColor = 5.0
    #nodes[1].wantedColor = -5.0
    #nodes[2].wantedColor = 0.0
    #nodes[3].wantedColor = 5.0

    graphs = allPossibleColorings(nodes)

    numIndividuals = 50
    population = initPop(numIndividuals, "")
    for generation in xrange(0,generations):
        print "Running generation ", generation
        for monkey in population:
            graphNumber = 0
            for a,b in itertools.product([0,1],[0,1]):
                #print "  Running individual ", id(monkey), " for input case ", [a,b]
                monkeyRun = runMonkey(cli, monkey, [a,b])
                #Color the graph in accordance with the buffers found,
                #and set the "wanted" color in the colored graph
                #like the current test case.
                coloredGraph = copy.deepcopy(graphs[graphNumber][0])
                color(coloredGraph, monkeyRun.recordings)
                monkey.graphs.append(coloredGraph)
                graphNumber += 1

            monkey.fitness = fitness(monkey)

        fitMonkeys = select(population)
        print "Most fit monkey of generation ", generation, " : ", fitMonkeys[0].fitness
        for g in fitMonkeys[0].graphs:
          print "-----------------"
          printColors(g, list())

        if(fitMonkeys[0].fitness < 12.0):
          print "-----------------------------------------"
          print "Succeeded. Fitness ", fitMonkeys[0].fitness
          print fitMonkeys[0].items
          print "-----------------------------------------"
          break
        #This should return a brand new set of monkeys
        population = breed(numIndividuals, fitMonkeys)
