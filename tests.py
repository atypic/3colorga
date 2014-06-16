from collections import namedtuple
import unittest
import ga


import emEvolvableMotherboard
from ttypes import *
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


class TestGA(unittest.TestCase):
    def setUp(self):
        #create directed graph to try on
        nodes = []
        for i in range(0,4):
            nodes.append(ga.Node(i, None, [], False))

        self.graph = nodes[0]
        nodes[0].neighbors = [nodes[1],nodes[2]]
        nodes[1].neighbors = [nodes[0],nodes[2],nodes[3]]
        nodes[2].neighbors = [nodes[0],nodes[1]]
        nodes[3].neighbors = [nodes[1]]

    def test_color(self):
        ga.color(self.graph, [[1],[1],[1],[1]])
        self.assertEqual(self.graph.neighbors[0].color, 1.0)

    def test_fitness(self):
        self.assertEqual(ga.fitness(self.graph, [[-5.0], [0.0], [5.0], [2.5]]), 45.0)

    def test_initPop(self):
        pop = ga.initPop(12,"")
        self.assertEqual(len(pop), 12)
        for i in pop:
            self.assertEqual(type(i.pin[0]),int)

#    def test_runPop(self):
#        transport = TSocket.TSocket('129.241.102.247', 9090)
#        transport = TTransport.TBufferedTransport(transport)
#
#        prot = TBinaryProtocol.TBinaryProtocol(transport)
#        cli = emEvolvableMotherboard.Client(prot)
#        transport.open()
#        recs = ga.runMonkey(cli, ga.initPop(1, "")[0])
#        for boo in recs:
#            self.assertTrue(len(boo.Samples) > 0)
#        transport.close()

    def test_ga(self):
        transport = TSocket.TSocket('129.241.102.247', 9090)
        transport = TTransport.TBufferedTransport(transport)
        prot = TBinaryProtocol.TBinaryProtocol(transport)
        cli = emEvolvableMotherboard.Client(prot)
        transport.open()
        cli.ping()
        ga.gaLoop(cli, 1)
        transport.close()


if __name__ == '__main__':
    unittest.main()
