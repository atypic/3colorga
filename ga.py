from __future__ import division
import sys

import pickle
import emEvolvableMotherboard
from ttypes import *
from collections import defaultdict

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random 
import copy
import itertools
import time

import numpy as np
import scipy
import scipy.stats as st
import scipy.spatial.distance as dist

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
    inputPin = -1    #Which of the inputs is this item
    graphNodeLabel = -1   #Which of the nodes is this item assigned to
    def __init__(self, it, ip, rp):
      self.item = it
      self.inputPin = ip
      self.graphNodeLabel = rp

class Monkey:
    """Thin wrapper for a monkey... individual"""
    items = list()
    recordings = {}
    fitness = -1.0
    graphs = list()

    def __init__(self, sch):
        self.items = sch;
        self.recordings = {}
        self.fitness = -1.0
        self.graphs = list()

#Take a graph and a list of buffers, get how fit the sample buffer is.
#Each buffer represents 1 node.
#The color is decided my the overall voltage level in the buffer.
#The sorting is such that node 0 is buffer 0 in the list of buffers.
def fitness2(monkey, numWork):
  s = 0
  voltColor = defaultdict(list)
  for g in monkey.graphs:
    getVoltColor(g, voltColor)
    #s += score(g)
  #print voltColor
  minSum = 0.0
  numMin = 0
  colors = []
  colorReps = defaultdict(list)
  #print voltColor
  for voltkey in voltColor.keys():
    for pair in itertools.combinations(voltColor[voltkey], r=2):
      #print "voltkey pair", pair[0], " bluh ", pair[1]
      minSum += abs(pair[0] - pair[1])/len(voltColor[voltkey])

    srt = sorted(voltColor[voltkey])
    rep = srt[0] + ((srt[-1] - srt[0])/2.0)
    colors.append(rep)
    colorReps[voltkey] = rep

  maxSum = 0.0
  numMax = 0
  #print colors
  for pair in itertools.combinations(colors, r=2):
    #print "Comparing", pair[0], " to ", pair[1]
    maxSum += abs(pair[0] - pair[1])
    numMax += 1


  if isWorkingDevice(monkey, colorReps, 0.1):
    numWork += 1
    print "Working device! :)"

  print "Color separation score: ", maxSum, "Per-color similarity score: ", minSum
  return maxSum - minSum
  #return maxSum

def fitness(monkey):
  s = 0
  voltColor = defaultdict(list)
  for g in monkey.graphs:
    getVoltColor(g, voltColor)
  #print voltColor
  maxim = 0.0
  numMin = 0
  fit = 0.0
  colors = []
  colorReps = defaultdict(list)

  #Separation
  fit += min(voltColor[3.0]) - max(voltColor[2.0])
  fit += min(voltColor[2.0]) - max(voltColor[1.0])

  #Green band tiny as possible in the middle
  fit -= max(voltColor[2.0]) - min(voltColor[2.0])

  numWork = 0
  if (min(voltColor[3.0]) > max(voltColor[2.0])) and (min(voltColor[2.0]) > max(voltColor[1.0])):
    print "Certified working device :-):"
    print voltColor
    numWork += 1
    fit *= 1.15
  else:
    fit *= 0.85

  return (fit, numWork)




def isWorkingDevice(monkey, colors, voltRange):
  r = True
  for g in monkey.graphs:
    nodeCorrectness = []
    nodeVisited = []
    _isWorkingDevice(g, colors, nodeVisited, nodeCorrectness, voltRange)
    if False in nodeCorrectness:
      r = False

  return r

def _isWorkingDevice(node, colors, visitedList, correctList, voltRange):
  if  node.label in visitedList:
    return;
  else:
    visitedList.append(node.label)
    if ((node.color + voltRange) < colors[node.wantedColor]) and ((node.color - voltRange) > colors[node.wantedColor]):
      print "dec:", node.color, "want:", colors[node.wantedColor]
      correctList.append(True)
    else:
      correctList.append(False)

    for nig in node.neighbors:
      _isWorkingDevice(nig, colors, visitedList, correctList, voltRange)



def score(node):
  """Score the subgraph from node.
  """
  subSum = 0.0
  if node.visited == True:
    return 0.0

  node.visited = True
  multiplier = 1.0
  if node.neighbors != []:
    for n in node.neighbors:
      #if (node.wantedColor > n.wantedColor) & (node.color > n.color):
      #  multiplier = 0.7
      #elif (node.wantedColor < n.wantedColor) & (node.color < n.color):
      #  multiplier = 0.7

      subSum += multiplier * score(n)

  s = abs(node.wantedColor - node.color) + subSum

  return s


def getVoltColor(node, voltColors):
  if node.visited == True:
    return voltColors

  node.visited = True
  voltColors[node.wantedColor].append(node.color)
  if node.neighbors != []:
    for n in node.neighbors:
      getVoltColor(n, voltColors)

  return voltColors


def distance(a, b):
    return abs(a.color - b.color)

""" Color also collects all colors in the passed coloring hash.

maxVolts[color] = (0.4, 0.3, 0.1)

We then proceed to analyze this array to find the fitness.
We want the difference in the list to be minimized, because it's the same color,
but the distance between each of the colors maximal.

So we sort the list, take max-min/2 to find the middle value, then 
we we check the difference between all pairs.
"""
def color(node, buffers):
    """Traverse graph depth first, color it as the buffers suggest"""
    """Buffer is indexed on node labels."""
    if node.color == None:
        if(len(buffers[node.label].Samples) != 0):
            #node.color = sum(samplesToVolts(buffers[node.label].Samples))/len(buffers[node.label].Samples)
            node.color = max(samplesToVolts(buffers[node.label].Samples))
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
  for item in monkey.items:
    print "  Item ", item.item.operationType, " on pin ", item.item.pin, " input pin", item.inputPin,\
        "graph node label", item.graphNodeLabel 
    if item.item.operationType == emSequenceOperationType().CONSTANT:
      print "    Voltage: ", item.item.amplitude
    elif item.item.operationType == emSequenceOperationType().DIGITAL:
      print "    Freq:", item.item.frequency, " Cycle:", item.item.cycleTime
  print "------------------"

def runMonkey(cli, monkey, inputValue):
    """Runs population, returns a list of recordings (full objects)"""

    cli.reset()
    cli.clearSequences()
    recpins = {}
    for w in monkey.items:
        """This is a input pin, digital one."""
        if w.inputPin != -1:
            w.item.frequency = inputValue[w.inputPin]
            w.item.cycleTime = 100

        elif w.item.operationType == emSequenceOperationType().RECORD:
            recpins[w.graphNodeLabel] = w.item.pin[0]

        #print w.item
        cli.appendSequenceAction(w.item)
    #set up the input pins.
    cli.runSequences()
    cli.joinSequences()
   
    for r in recpins.keys():
        monkey.recordings[r] = cli.getRecording(recpins[r])

    return monkey


def diff(m1, m2):

  v1 = 16 * [0]
  v2 = 16 * [0]

  for its in m1.items:
    i = its.item
    if i.operationType == emSequenceOperationType().CONSTANT:
      v1[i.pin[0]] = i.amplitude
    elif i.operationType == emSequenceOperationType().DIGITAL:
      v1[i.pin[0]] = i.frequency*(i.cycleTime/100.0)
    elif i.operationType == emSequenceOperationType().RECORD:
      v1[i.pin[0]] = 1.0

  for its in m2.items:
    i = its.item
    if i.operationType == emSequenceOperationType().CONSTANT:
      v2[i.pin[0]] = i.amplitude
    elif i.operationType == emSequenceOperationType().DIGITAL:
      v2[i.pin[0]] = (i.frequency/10000.0)*(i.cycleTime/100.0)
    elif i.operationType == emSequenceOperationType().RECORD:
      v2[i.pin[0]] = 1.0

  return dist.cosine(v1,v2)

def initPop(size, allowedOpTypes):
    """Initializes a list of Monkey objects, len == size; each Monkey based on allowed_types."""
    population = []
    for ind in xrange(0,size):
        #print " -New individual - "
        individual = []
        #limits.
        sequenceRunTimeMs = 50
        maxPins = 16
        numDA = 0
        maxDA = 8
        unusedPins = set(range(0,maxPins))
        numNodes = 4
        numInputPins = 4

        usedDACchannels = 0
        #First the recording pins for this individual
        for j in xrange(0,numNodes):
            foo = random.choice(list(unusedPins))
            unusedPins.remove(foo)

            it = emSequenceItem()
            it.pin = [foo]
            it.startTime = 0
            it.endTime = sequenceRunTimeMs
            it.frequency = 10000
            it.operationType = emSequenceOperationType().RECORD   #implies analogue 
            individual.append(Item(it, -1, j))

        for i in xrange(0, numInputPins):
            it = emSequenceItem()
            foo = random.choice(list(unusedPins))
            unusedPins.remove(foo)
            it.pin = [foo]
            it.startTime = 10
            it.endTime = sequenceRunTimeMs + 5
            it.operationType = emSequenceOperationType().DIGITAL
            individual.append(Item(it, i, -1))

        #Use remaining pins for DA channels.
#        for d in range(0, maxPins - numNodes - numInputPins):
        for d in range(0, 8):
            it = emSequenceItem()

            pin = random.sample(unusedPins, 1)
            unusedPins.remove(pin[0])
            it.pin = pin
            print "PIN:", pin
            it.startTime = 0
            it.endTime = sequenceRunTimeMs
            
            opType = random.choice(allowedOpTypes)
            it.operationType = opType

            if opType == emSequenceOperationType().CONSTANT:
                it.amplitude = random.randrange(128,200)
                #it.amplitude = random.choice([2,255])
            elif opType == emSequenceOperationType().DIGITAL:
                it.frequency = random.randint(0,1000000)
                it.cycleTime = random.randint(0,100)
            
            individual.append(Item(it, -1, -1))

        newMonkey = Monkey(individual)

        population.append(newMonkey)

    return population

def selectAndBreed(pop, popSize):
    """Select uses tournament selections. Population is a list of Monkeys"""
    """len(pop)/2 tournaments are held."""
    newChildren = []
    while(len(newChildren) < popSize):
      numParents = 2
      winners = []
      for i in xrange(0,numParents):
        fighters = random.sample(pop, 2)
        sortedByFitness = sorted(fighters, key=lambda m: m.fitness, reverse=True)
        
        if random.random() < 0.9:
          winners.append(sortedByFitness[0])
        else:
          winners.append(sortedByFitness[1])
        
      newChildren.extend(breed(winners))
      #winners = sorted(winners, key=lambda m: m.fitness, reverse=False)

    return newChildren

#Breed a new population based on passed population
#Always returns two new kids
def breed(parents):
  bredPopulation = []
  while(len(bredPopulation) < 2):
    child = Monkey(copy.deepcopy(parents[0].items))
    child2 = Monkey(copy.deepcopy(parents[1].items))

    j = 0
    xpoint = random.randrange(0,16)
    pinToItem1 = {}
    pinToItem2 = {}
    for i,i2 in zip(child.items, child2.items):
      it = i.item
      it2 = i2.item
      #Note that this allows the same pin to appear more than one time
      if j > xpoint:
        tmp = it.pin
        it.pin = it2.pin
        it2.pin = tmp
        #print it2.pin, " moved to ", it.pin
        #print it.pin, " moved to ", it2.pin
      else:
        tmp = it2.pin
        it2.pin = it.pin
        it.pin = tmp
        #print it.pin, " moved to ", it2.pin
        #print it2.pin, " moved to ", it.pin

      j += 1

      pinToItem1[it.pin[0]] = it
      pinToItem2[it2.pin[0]] = it2

   
    for c,pinToItem in zip([child.items, child2.items],[pinToItem1, pinToItem2]):
      for q in c:
        i = q.item
        #First mutation
        dice = random.random()
        dice2 = random.random()
        mutationRate = 0.90

        #Two mutation possibilities
        if dice2 > 0.5:
          if i.operationType == emSequenceOperationType().CONSTANT:
            if dice > mutationRate:
              i.amplitude = int(max(128,min(230, i.amplitude + random.gauss(0,15))))
              #print "new amp", i.amplitude
          elif i.operationType == emSequenceOperationType.DIGITAL:
            if dice > mutationRate:
              i.frequency = max(1, min(75000000, i.frequency + int(random.gauss(0,100000))))
              i.cycleTime = int(min(100,max(0, i.cycleTime + random.gauss(0,20))))
        else:
          #Second mutation operator: internal pin increment (or swap, if to-pin in use)
          if dice > mutationRate:
            toPin = (i.pin[0] + 1)%16
            fromPin = i.pin[0]

            if pinToItem.has_key(toPin):
              tmp = copy.copy(pinToItem[fromPin].pin)
              pinToItem[fromPin].pin = pinToItem[toPin].pin
              pinToItem[toPin].pin = tmp
            else:
              i.pin = [toPin]


    if isValidMonkey(child):
      bredPopulation.append(child)
    if isValidMonkey(child2):
      bredPopulation.append(child2)

  #for m in bredPopulation:
  #  print "------"
  #  for j in m.items:
  #    print j.item
  #  print "------"
  return bredPopulation

def isValidMonkey(monkey):
  uniqueRecPins = []
  for it in monkey.items:
    #if it.item.operationType == emSequenceOperationType().RECORD:
    if it.item.pin in uniqueRecPins:
      return False
    else:
      uniqueRecPins.append(it.item.pin)

  #if len(uniqueRecPins) != 4:
  #return False

  return True

def samplesToVolts(buf):
  return [i * (5.0/4096.0) for i in buf]

def allPossibleColorings(graph):
  R = 1.0
  G = 2.0
  B = 3.0

  ret = []
  g = copy.deepcopy(graph)
  g[0].wantedColor = B
  g[1].wantedColor = G
  g[2].wantedColor = R
  g[3].wantedColor = R
  ret.append(g)

  g = copy.deepcopy(graph)
  g[0].wantedColor = G
  g[1].wantedColor = B
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

    numIndividuals = 65

    populations = [initPop(numIndividuals, [emSequenceOperationType().CONSTANT, emSequenceOperationType().DIGITAL]),\
                  initPop(numIndividuals, [emSequenceOperationType().CONSTANT]),\
                  initPop(numIndividuals, [emSequenceOperationType().DIGITAL])]
    popNames = ["Mixed", "Static voltages", "Square waves"]
    popNum = 0

    avgFitness = defaultdict(list)
    maxFitness = defaultdict(list)
    numWorking = defaultdict(list)
    diffs = defaultdict(list)

    totalNumGenerations = generations * len(populations)
    runGenerations = 0

    for popName, population in zip(popNames,populations):
      cli.reset()

      for generation in xrange(0,generations):
          print "Running generation ", generation
          genStartTime = time.time()
          numWorkingInGeneration = 0
          avgDiffPerGeneration = 0
          numComps = 0
          for monkey in population:
              for monkeyPair in itertools.combinations(population, r=2):
                avgDiffPerGeneration += diff(monkeyPair[0], monkeyPair[1])
                numComps += 1
              

              graphNumber = 0
  #           for a,b in itertools.product([0,1],[0,1]):
              for case in [[0,0,0,0],[1,1,1,1]]:
              #for case in [[0,0]]:
                  #print "  Running individual ", id(monkey), " for input case ", [a,b]
                  monkeyRun = runMonkey(cli, monkey, case)
                  #Color the graph in accordance with the buffers found,
                  #and set the "wanted" color in the colored graph
                  #like the current test case.
                  coloredGraph = copy.deepcopy(graphs[graphNumber])
                  #print "ID of recordings to be used for coloring: ", id(monkeyRun.recordings)
                  color(coloredGraph[0], monkeyRun.recordings)
                  monkey.graphs.append(copy.deepcopy(coloredGraph[0]))
                  graphNumber += 1
              
              fitWork = fitness(monkey)
              monkey.fitness = fitWork[0]
              numWorkingInGeneration += fitWork[1]


          #Statistics per generation
          fitMonkeys = sorted(population, key=lambda m: m.fitness, reverse=True)
          print "Most fit monkey of generation ", generation, " : ", fitMonkeys[0].fitness
          printMonkey(fitMonkeys[0])
          avg = sum([a.fitness for a in fitMonkeys])/len(fitMonkeys)
          print "Arithmetic average fitness", avg, " for ", popName
          print "Working devices:", numWorkingInGeneration

          maxFitness[popName].append(fitMonkeys[0].fitness)
          avgFitness[popName].append(avg)
          numWorking[popName].append(numWorkingInGeneration)
          diffs[popName].append(avgDiffPerGeneration/numComps)

          for g in fitMonkeys[0].graphs:
            print "-----------------" 
            printColors(g, list())
         # if(fitMonkeys[0].fitness < .0):
         #   print "-----------------------------------------"
         #   print "Succeeded. Fitness ", fitMonkeys[0].fitness
         #   print fitMonkeys[0].items
          #  print "-----------------------------------------"
          #  break
          #This should return a brand new set of monkeys
          
          population = selectAndBreed(population, numIndividuals)
          print "Bred new population of ", len(population), " monkeys."
          genEndTime = time.time()

          runGenerations += 1
          genTime = (genEndTime - genStartTime)
          totTimeLeft = (totalNumGenerations - runGenerations) * genTime
          print "There are ", (totTimeLeft)/60, "minutes of runtime left."
      popNum += 1


    
    pickles = [maxFitness, avgFitness, numWorking, diffs]
    out = open('plotdata.pkl', 'wb')
    pickle.dump(pickles, out)
    out.close()

    plt.figure()
    plt.title("Max fitness")
    #plt.ylim(-5,5)
    for g in popNames:
      plt.plot(maxFitness[g], label=g)
    plt.legend(loc=4)
    plt.savefig("plot.pdf")
    plt.close()


    plt.figure()
    plt.title("Arithmetic average fitness")
    #plt.ylim(-5,5)
    for g in popNames:
      plt.plot(avgFitness[g], label=g)
    plt.legend(loc=4)
    plt.savefig("avg.pdf")
    plt.close()

    plt.figure()
    plt.title("Working devices")
    for g in popNames:
      plt.plot(numWorking[g], label=g)
    plt.ylim(ymin=0)
    plt.legend(loc=4)
    plt.savefig("working.pdf")
    plt.close()

    plt.figure()
    plt.title("Cosine distance")
    #normalize the differences to 0 and 1
    plt.ylim(0, 1.0)
    for g in popNames:
      #maxDiff = max(diffs[g])
      #normDiff = []
      #for d in diffs[g]:
        #normDiff.append((d/maxDiff))
      plt.plot(diffs[g])

    plt.savefig("diffs.pdf")
    plt.close()
