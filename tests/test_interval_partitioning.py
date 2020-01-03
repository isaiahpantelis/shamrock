import os
import sys
from time import process_time
import numpy as np
import matplotlib.pyplot as plt
import math
import unittest
import random

folder_containing_shamrock_package = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(folder_containing_shamrock_package)
# print(f'-- {folder_containing_shamrock_package}')

import shamrock as sh


# ----------------------------------------------------------------------------------------------------------------------


class TestPartitionClass(unittest.TestCase):

    def test_default_constructor(self):
        p = sh.Partition()
        # print(f'\n\n{p}\n\n')
        self.assertEqual(len(p.x), 0)

    def test_equal_end_points(self):
        a = 0.0
        b = 0.0
        K = 1.0
        P = sh.Partition(a, b, K)
        self.assertEqual(len(P.x), 1)

    def test_b_less_than_a(self):
        a = 0.0
        b = -1.0
        K = 1.0
        P = sh.Partition(a, b, K)
        self.assertEqual(len(P.x), 0)

    def test_negative_K(self):
        a = -1.0
        b = 1.0
        K = -1.0
        # -- Should raise an OverflowError because the value -1.0 cannot be converted to unsigned int.
        with self.assertRaises(OverflowError):
            sh.Partition(a, b, K)

    def test_zero_K(self):
        a = -abs(random.random())
        b = abs(random.random())
        K = 0.0
        p = sh.Partition(a, b, K)
        # print(f'\n\n{p}\n\n')
        # print(p.x[0] - a)
        self.assertEqual(len(p.x), 2)
        self.assertLessEqual(abs(p.x[0] - a), 1e-12)
        self.assertLessEqual(abs(p.x[1] - b), 1e-12)

    def test_refining_the_empty_partition(self):
        a = 1.0
        b = -1.0
        K = 3
        p = sh.Partition(a, b, K)
        self.assertEqual(len(p.x), 0)
        p.refine()
        self.assertEqual(len(p.x), 0)

    def test_refining_a_singleton(self):
        a = 1.0
        b = 1.0
        K = 0
        p = sh.Partition(a, b, K)
        # print(p)
        self.assertEqual(len(p.x), 1)
        p.refine()
        self.assertEqual(len(p.x), 1)
        # print(p)

    def test_number_of_points_of_refinement(self):
        a = -abs(random.random())
        b = abs(random.random())
        K = abs(random.randint(0, 15))
        p = sh.Partition(a, b, K)
        self.assertEqual(p.K, K)
        self.assertEqual(p.N, 2 ** K)
        p.refine()
        self.assertEqual(p.K, K + 1)
        self.assertEqual(p.N, 2 ** (K + 1))

    def test_unbounded_interval_01(self):
        a = -np.inf
        b = abs(random.random())
        K = abs(random.randint(0, 15))
        with self.assertRaises(NotImplementedError, msg=f'[{a}, {b}]'):
            p = sh.Partition(a, b, K)

    def test_unbounded_interval_02(self):
        a = -abs(random.random())
        b = np.inf
        K = abs(random.randint(0, 15))
        with self.assertRaises(NotImplementedError, msg=f'[{a}, {b}]'):
            p = sh.Partition(a, b, K)

    def test_unbounded_interval_03(self):
        a = -np.inf
        b = np.inf
        K = abs(random.randint(0, 15))
        with self.assertRaises(NotImplementedError, msg=f'[{a}, {b}]'):
            p = sh.Partition(a, b, K)

    def test_unbounded_interval_04(self):
        a = -np.inf
        b = -np.inf
        K = abs(random.randint(0, 15))
        p = sh.Partition(a, b, K)
        self.assertEqual(len(p.x), 0, msg=f'[{a}, {b}]')

    def test_unbounded_interval_05(self):
        a = np.inf
        b = np.inf
        K = abs(random.randint(0, 15))
        p = sh.Partition(a, b, K)
        self.assertEqual(len(p.x), 0, msg=f'[{a}, {b}]')

    def test_unbounded_interval_06(self):
        a = np.inf
        b = -np.inf
        K = abs(random.randint(0, 15))
        p = sh.Partition(a, b, K)
        self.assertEqual(len(p.x), 0, msg=f'[{a}, {b}]')

    def test_successive_refinement_and_coarsening(self):
        a = -abs(random.random())
        b = abs(random.random())
        K = abs(random.randint(0, 15))
        p = sh.Partition(a, b, K)
        x_old = [_ for _ in p.x]
        self.assertEqual(2 ** K + 1, len(p.x))
        p.refine()
        self.assertEqual(2 ** (K + 1) + 1, len(p.x))
        p.coarsen()
        self.assertEqual(2 ** K + 1, len(p.x))
        x_new = [_ for _ in p.x]

        for old, new in zip(x_old, x_new):
            self.assertEqual(old, new)

    def test_coarsening_the_empty_partition(self):
        a = 1.0
        b = -1.0
        K = 3
        p = sh.Partition(a, b, K)
        self.assertEqual(len(p.x), 0)
        p.coarsen()
        self.assertEqual(len(p.x), 0)

    def test_coarsening_a_singleton(self):
        a = 1.0
        b = 1.0
        K = 0
        p = sh.Partition(a, b, K)
        # print(p)
        self.assertEqual(len(p.x), 1)
        p.coarsen()
        self.assertEqual(len(p.x), 1)
        # print(p)

    def test_coarsening_a_two_point_partition(self):
        a = -1.0
        b = 1.0
        K = 0
        p = sh.Partition(a, b, K)
        # print(p)
        self.assertEqual(len(p.x), 2)
        p.coarsen()
        self.assertEqual(len(p.x), 2)
        # print(p)

    # -- From the documentation to use as templates
    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')
    #
    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)


# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    unittest.main()
