from random import * # type: ignore
from math import * #type: ignore

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
        return 0
    else: 
        tempVariance = sum((x - inputMean) ** 2 for x in inputList)

        if isSample == True:
            return (tempVariance / (n - 1))
        else:
            return (tempVariance / n)    
    
# Input a list or tuple and whether or not it is a sample (if no input is given, it will be treated as a population), and return the standard deviation of the list.
def stDev(inputList, isSample : bool = False):
        return(variance(inputList, isSample) ** 0.5)

# Input a list or tuple, and return a list of the z-scores of every value of the list. If an index is given, it will return the z-score of the given index.
def zScorify(inputList: list | tuple, index: float = None, sample: bool = False):
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

if __name__ == "__main__":
    print("running script:")