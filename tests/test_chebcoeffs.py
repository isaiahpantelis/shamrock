import os
import sys
from time import process_time
import numpy as np
import matplotlib.pyplot as plt
import math
import unittest
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shamrock as sh
import example_functions as exf


# ----------------------------------------------------------------------------------------------------------------------


class TestChebcoeffs(unittest.TestCase):

    def test_something(self):
        pass


# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    unittest.main()
