from random import *
from math import *
from typing import *

probability = Annotated[float, lambda p: 0 <= p <= 1]

def validateProbability(func):
    # Validates that probabilities are between 0 and 1
    def wrapper(p, *args, **kwargs):
        if not (0 <= p <= 1):
            raise ValueError("p must be between 0 and 1.")
        return func(p, *args, **kwargs)
    return wrapper

# Input a list or tuple, return a mean.
def mean(inputList : list | tuple):
    if not inputList:
         raise ZeroDivisionError("Cannot find the mean of an empty set; division by zero.")
    elif not all(isinstance(x, (int, float)) for x in inputList):
         raise ValueError("Must be a list of numerical values.")
    elif any(isnan(x) or isinf(x) for x in inputList):
         raise ValueError("List cannot contain indeterminate (invalid) values NaN or Inf.")
    else:
        return sum(inputList) / len(inputList)

# Input a list or tuple and whether or not it is a sample (if no input is given, it will be treated as a population), and return the variance of the list.
def variance(inputList: list | tuple, isSample: bool = False):
    tempVariance = 0
    n = len(inputList)
    inputMean = mean(inputList)

    if not inputList:
         raise ZeroDivisionError("Cannot find the mean of an empty set; division by zero.")
    elif n == 1:
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
def stDev(inputList, isSample : bool = True):
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
        
# Means and standard deviations of various distribution types.
class dists:

# Mu is another way to write "the population mean"
# Sigma is another way to write "the population standard deviation"

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
        
        # The average distance from
        @staticmethod
        @validateProbability
        def stDev(p: probability):
            if p == 0:
                raise ZeroDivisionError("For geometric distributions, p cannot equal zero. Success can never be reached.")
            else:
                return(((1-p) ** 0.5) / p)

if __name__ == "__main__":
    print("running script:")

    print(dists.binom.stDev(0.5,5))