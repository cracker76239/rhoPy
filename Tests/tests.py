import sys
sys.path.append('C:/Users/caleb/Documents/GitHub/rhoPy')

from random import *
import pytest
from rhoPy.rho import * # type: ignore

# Some tests.

def testMeanEmpty():
    with pytest.raises(ZeroDivisionError):
        mean([])

def testMeanNumerical():
    with pytest.raises(ValueError):
        mean([1,2,"three"])

def testMean():
    assert mean([1,2,3,4,5]) == 3
    
def testFMean():
    assert mean([1,2,3,4,5]) == 3

def testMeanWeighted():
    assert mean([1,2,3,4,5],[0.2,0.2,0.2,0.2,0.2]) == 3

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

def testSampleWithReplacement():
    result = sample([1, 2, 3, 4, 5], 3, replacement=True)
    assert len(result) == 3
    assert all(item in [1, 2, 3, 4, 5] for item in result)

def testSampleWithoutReplacement():
    result = sample([1, 2, 3, 4, 5], 3, replacement=False)
    assert len(result) == 3
    assert len(set(result)) == 3  # All elements should be unique (no replacement)
    assert all(item in [1, 2, 3, 4, 5] for item in result)

def testSampleMoreThanListSize():
    result = sample([1, 2, 3], 5, replacement=True)
    assert len(result) == 5

def testSamplingPropStDev():
    assert dists.sampling.prop.stDev(0.5, 10) == pytest.approx(0.1581, rel=1e-2)
    assert dists.sampling.prop.stDev(0.2, 20) == pytest.approx(0.0894, rel=1e-2)

def testSamplingMeanStDev():
    assert dists.sampling.mean.stDev(10, 5) == pytest.approx(4.4721, rel=1e-2)
    assert dists.sampling.mean.stDev(2, 10) == pytest.approx(0.6325, rel=1e-2)

def testBinom():
    assert dists.binom.mean(0.5, 10) == 5
    assert dists.binom.mean(0.3, 20) == 6
    assert dists.binom.stDev(0.5, 10) == pytest.approx(1.58, rel=1e-2)
    assert dists.binom.stDev(0.2, 20) == pytest.approx(1.79, rel=1e-2)

def testGeomet():
    assert dists.geomet.mean(0.5) == 2
    assert dists.geomet.mean(0.25) == 4
    assert dists.geomet.stDev(0.5) == (0.5**0.5) / 0.5
    assert dists.geomet.stDev(0.25) == (0.75**0.5) / 0.25

def testProbabilityValidation():
    with pytest.raises(ValueError):
        dists.geomet.mean(69)

def testMedian():
    assert median([1,2,3,4,5]) == 3
    assert median([1,2,3,4]) == 2.5
    
def testQuickMedian():
    assert quickMedian([1,2,3,4,5]) == 3
    assert quickMedian([1,2,3,4]) == 2.5
    
def testQuartiles():
    list = [1,2,3,4,5]
    assert Q1(list) == 1.5
    assert Q3(list) == 4.5
    assert IQR(list) == 3

def test_chisq():
    assert dstr.chisqrdist([[1,2,3],[4,5,6],[7,8,9]]) == [[1.6, 2.0, 2.4], [4.0, 5.0, 6.0], [6.4, 8.0, 9.6]]
    assert dstr.chisqrdist([1,2,3],[2,3,4]) == [2,3,4]
    assert dstr.chisqrdist([1,2,3]) == None