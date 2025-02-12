import sys
sys.path.append('./statsLib')

from random import *
import pytest # type: ignore
from statsLib import * # type: ignore

# Some tests.

def testMeanEmpty():
    with pytest.raises(ZeroDivisionError):
        mean([])

def testMeanNumerical():
    with pytest.raises(ValueError):
        mean([1,2,"three"])

def testMean():
    assert mean([1,2,3,4,5]) == 3

def testVarianceEmpty():
    with pytest.raises(ZeroDivisionError):
        variance([])

def testVarianceZero():
    assert variance([1]) == 0

def testVariance():
    assert variance([1,2,3,4,5]) == 2

# If variance raises all the right errors, stDev doesn't have to worry about that.
def testStDev():
    assert stDev([1,2,3,4,5]) == 2 ** 0.5

def testZscorifyList():
    assert zScorify([1,2,3,4,5]) == [-1.41, -0.71, 0, 0.71, 1.41]

def testZscorifyIndex():
    assert zScorify([1,2,3,4,5],2) == 0


        



    





