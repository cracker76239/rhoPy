from random import *
from math import *
from typing import Annotated

probability = Annotated[float, lambda p: 0 <= p <= 1]

# Quickselect algorithm for median functions or others which require sorted lists.
def quickselect(inputList, k):
    pivot = random.choice(inputList)
    
    lows = [x for x in inputList if x < pivot]
    highs = [x for x in inputList if x > pivot]
    pivots = [x for x in inputList if x == pivot]
    
    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots))

def validateProbability(func):
    # Validates that probabilities are between 0 and 1.
    # Only works for individual values, not contents of a set. I don't know how to do the latter.
    def wrapper(p, *args, **kwargs):
        if not (0 <= p <= 1):
            raise ValueError("p must be between 0 and 1.")
        return func(p, *args, **kwargs)
    return wrapper

# Input a set of numbers, return a mean.
# If a set of weights is input, it will return the mean based on those weights.
def mean(inputList : list | tuple, inputWeights: list[probability] | tuple[probability] = None):
    if not inputList:
        raise ZeroDivisionError("Cannot find the mean of an empty set; division by zero.")
    elif (not all(isinstance(x, (int, float)) for x in inputList) or (any(isnan(x) or isinf(x) for x in inputList))):
        raise ValueError("Must be a list of numerical and determinate values. A string, bool, NaN, etc. was input.")
    elif inputWeights == None:
        return sum(inputList) / len(inputList)
    else:
        if len(inputList) != len(inputWeights):
            raise IndexError("The length of both the input set and weights must be equal.")
        if not all(x > 0 for x in inputWeights):
            raise ValueError("Weights must be non-negative.")
        total_weight = sum(inputWeights)
        if total_weight != 1:
            raise ZeroDivisionError("The sum of weights cannot be zero.")
        return sum(inputList[i] * inputWeights[i] for i in range(len(inputList))) / total_weight

# Same as mean(), but uses floats for more precision.
# Also allows NaN and inf to be input.
def fmean(inputList : list | tuple):
    if not inputList:
        raise ZeroDivisionError("Cannot find the mean of an empty set; division by zero.")
    elif (not all(isinstance(x, (int, float)) for x in inputList)):
        raise ValueError("Must be a list of numerical. A string, bool, etc. was input.")
    else:
        return float(sum(inputList)) / float(len(inputList))

# Input a set of numbers, return a geometric mean.
# If a set of weights is input, it will return the geometric mean based on those weights.
def geometMean(inputList: list | tuple, inputWeights: list[probability] | tuple[probability] = None):
    if not inputList:
        raise ZeroDivisionError("Cannot find the geometric mean of an empty set; division by zero.")
    elif not all(isinstance(x, (int, float)) and x > 0 for x in inputList):
        raise ValueError("Geometric mean requires a list of positive numerical values.")
    
    if inputWeights == None:
        return prod(inputList) ** (1 / len(inputList))
    else:
        if sum(inputWeights) != 1:
            raise ValueError("The weights for a set of values must sum to one.")
        elif len(inputList) != len(inputWeights):
            raise IndexError("The length of both the input set and its weights must be equal.")
        else:
            return prod(inputList[i] ** inputWeights[i] for i in range(len(inputList)))


# Input a set of numbers, return a harmonic mean.
# If a set of weights is input, it will return the harmonic mean based on those weights.
def harmMean(inputList: list | tuple, inputWeights: list[probability] | tuple[probability] = None):
    if not inputList:
        raise ZeroDivisionError("Cannot find the harmonic mean of an empty set; division by zero.")
    elif not all(isinstance(x, (int, float)) and x > 0 for x in inputList):
        raise ValueError("Harmonic mean requires a list of positive numerical values.")
    
    if inputWeights == None:
        return len(inputList) / sum(1 / x for x in inputList)
    else:
        
        if sum(inputWeights) != 1:
            raise ValueError("The weights for a set of values must sum to one.")
        elif len(inputList) != len(inputWeights):
            raise IndexError("The length of both the input set and its weights must be equal.")
        else:
            return sum(inputWeights) / sum(inputWeights[i] / inputList[i] for i in range(len(inputList)))


# Input a list or tuple and whether or not it is a sample (if no input is given, it will be treated as a population), and return the variance of the list.
def variance(inputList: list | tuple, isSample: bool = False):
    tempVariance = 0
    n = len(inputList)
    inputMean = mean(inputList)

    if not inputList:
         raise ZeroDivisionError("Cannot find the mean of an empty set; division by zero.")

    if n == 1:
        return 0 # The variance and standard deviation of a set containing one number is zero.
    else: 
        tempVariance = sum((x - inputMean) ** 2 for x in inputList)

        if isSample == True:
            return (tempVariance / (n))
        else:
            return (tempVariance / n)    
    
# Input a list or tuple and whether or not it is a sample (if no input is given, it will be treated as a population), and return the standard deviation of the list.
def stDev(inputList, isSample : bool = False):
        return(variance(inputList, isSample) ** 0.5)

# Returns the z-score of a number.
def zScore(value, mean, stDev):
    return((value - mean) / stDev)

# Input a list or tuple, and return a list of the z-scores of every value of the list. If an index is given, it will return the z-score of the given index.
def zScorify(inputList: list | tuple, index: int = None, sample: bool = False):
    inputMean = mean(inputList)
    inputStDev = stDev(inputList, sample)

    if index == None:
        return[round(((x - inputMean) / inputStDev), 2) for x in inputList]
    else:
       return((inputList[index] - inputMean) / inputStDev)

# Input a list or tuple, and return a sample of size n with or without replacement.
def sample(inputList: list | tuple, n: int, replacement: bool = True):
    popSize = len(inputList)
    if replacement == True:
        return[inputList[randint(0,popSize - 1)] for _ in range(n)]
    else:
        tempList = inputList # Don't want to edit the real list!

        for i in range(n): #Puts a random value in the set into the first n indices.
            j = randint(0,popSize - 1) # A random index j corresponding to the ordered index i. 
            tempList[i], tempList[j] = tempList[j], tempList[i] # Puts the jth element in the ith spot. Ends up with n random elements sampled without replacement.
        return(tempList[:n])

# Sorts a list (O(n log n)), and returns the median.
def median(inputList: list | tuple):
    sortedList = sorted(inputList)
    length = len(inputList)

    if length % 2 == 0:
        mid = length // 2
        return((sortedList[mid] + sortedList[mid-1]) / 2)

    else:
        return sortedList[length // 2]

# Returns the median quickly using the quickselect algorithm.
# Best case: O(n) Worst case: O(n^2)
def quickmedian(inputList: list | tuple):
    length = len(inputList)
    if length % 2 == 0:
        return quickselect(inputList, length // 2)
    
    else:
        return((quickselect(inputList, length // 2 - 1) + quickselect(inputList, length // 2)) / 2)
    
# If i'm being honest, I don't really know what this is supposed to do.
# But, it returns the median of some frequencies and intervals and stuff.
def grouped_median(classIntervals: list[tuple | list], frequencies: list[int]):
    if len(classIntervals) != len(frequencies):
        raise IndexError("The number of class intervals must match the number of frequencies.")
    
    # Calculate cumulative frequency
    cumulativeFreq = [sum(frequencies[:i+1]) for i in range(len(frequencies))]
    total = sum(frequencies)

    if total == 0:
        raise ZeroDivisionError("Total frequency cannot be zero.")

    half = total / 2

    # Find median class
    for i, freqSum in enumerate(cumulativeFreq):
        if freqSum >= half:
            medianClassIndex = i
            break

    # Median class parameters
    L = classIntervals[medianClassIndex][0]  # Lower boundary
    F = cumulativeFreq[medianClassIndex - 1] if medianClassIndex > 0 else 0  # Cumulative freq before median class
    fm = frequencies[medianClassIndex]  # Frequency of median class
    h = classIntervals[medianClassIndex][1] - classIntervals[medianClassIndex][0]  # Class width

    # Grouped median formula
    median = L + ((half - F) / fm) * h

    return median

# Low Median: Median of the lower half of the sorted list
def lowMedian(inputList: list | tuple):
    sortedList = sorted(inputList)
    mid = len(sortedList) // 2

    # For an odd-length list, we exclude the middle element
    lower_half = sortedList[:mid] if len(sortedList) % 2 == 0 else sortedList[:mid]
    
    return median(lower_half)

# High Median: Median of the upper half of the sorted list
def highMedian(inputList: list | tuple):
    sortedList = sorted(inputList)
    mid = len(sortedList) // 2

    # For an odd-length list, we exclude the middle element
    upper_half = sortedList[mid+1:] if len(sortedList) % 2 == 1 else sortedList[mid:]

    return median(upper_half)
        
# Means and standard deviations of various distribution types.
class dists:

# Mu is another way to write "the population mean"
# Sigma is another way to write "the population standard deviation"
# Rho is another way to write "the population proportion"

    class sampling:
        
        # The sampling distribution of p-hat
        class prop:
    
            # Mean of the sampling distribution of p-hat is equal to p.
            
            # Returns the standard deviation of the sampling distribution of p-hat for sample size n.
            # Here, probability is used do ensure that the proportion is between 0 and 1. It is not an actual probability.
            @staticmethod
            @validateProbability
            def stDev(p: probability, n: int):
                return(( (p * (1 - p)) / n) ** 0.5)
            
        # The sampling distribution of x-bar
        class mean:
            
            # Mean of the sampling distribution of x-bar is equal to the population mean.
        
            # Returns the standard deviation of the sampling distribution of x-bar for sample size n.
            @staticmethod
            def stDev(sigma: float | int, n: int):
                return(sigma / (n ** 0.5))
    
    # Describes the probability of x successes given n attempts and p probability of success.
    class binom:
        
        # The mean of the probability distribution, A.K.A. the expected amount of successes.
        @staticmethod
        @validateProbability
        def mean(p: probability, n: int):
            return(n * p)
            
        # The standard deviation of the probability distribution.
        @staticmethod
        @validateProbability
        def stDev(p: probability, n: int):
            return((n * p * (1 - p)) ** 0.5)
    
    # Describes the probability of taking x amount of attempts to get a success for p probability of success.
    class geomet:
        
        # The expected amount of attempts for a success.
        @staticmethod
        @validateProbability
        def mean(p: probability):
            if p == 0:
                raise ZeroDivisionError("For geometric distributions, p cannot equal zero. Success can never be reached.")
            else:
                return(1/p)
        
        # The standard deviation of the probability distribution.
        @staticmethod
        @validateProbability
        def stDev(p: probability):
            if p == 0:
                raise ZeroDivisionError("For geometric distributions, p cannot equal zero. Success can never be reached.")
            else:
                return(((1-p) ** 0.5) / p)
            


if __name__ == "__main__":
    print("running script:")