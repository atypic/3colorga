import unittest
import main
from collections import namedtuple

class TestGA(unittest.TestCase):
    def setUp(self):
        Graph = namedtuple("Graph", "color neighbors")

    def test_fitness(self):

if __name__ == '__main__':
    unittest.main()
