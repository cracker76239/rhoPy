from random import *
from math import *
from typing import Annotated

probability = Annotated[float, lambda p: 0 <= p <= 1]
comparison = Annotated[str, "<" or "=" or ">"]

# Quickselect algorithm for median functions or others which require sorted lists.
def quickSelect(inputList, k):
    pivot = choice(inputList)
    
    lows = [x for x in inputList if x < pivot]
    highs = [x for x in inputList if x > pivot]
    pivots = [x for x in inputList if x == pivot]

    if k < len(lows):
        return quickSelect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivots[0]
    else:
        return quickSelect(highs, k - len(lows) - len(pivots))

def validateProbability(func):
    def wrapper(p, *args, **kwargs):
        # Handle list or scalar input
        if isinstance(p, list):
            if not all(isinstance(val, (int, float)) and 0 <= val <= 1 for val in p):
                raise ValueError("All probabilities in the list must be between 0 and 1.")
        elif not isinstance(p, (int, float)) or not (0 <= p <= 1):
            raise ValueError("p must be a number between 0 and 1.")
        
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
        raise ValueError("Must be a list of numerical values. A string, bool, etc. was input.")
    else:
        return float(sum(inputList)) / float(len(inputList))

# Input a set of numbers, return a geometric mean.
# If a set of weights is input, it will return the geometric mean based on those weights.
@validateProbability
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
            return (tempVariance / (n - 1))
        else:
            return (tempVariance / n)    
    
# Input a list or tuple and whether or not it is a sample (if no input is given, it will be treated as a population), and return the standard deviation of the list.
def std(inputList, isSample : bool = False):
        return(variance(inputList, isSample) ** 0.5)

# Returns the z-score of a number.
def zScore(value, mean, std):
    return((value - mean) / std)

# Input a list or tuple, and return a list of the z-scores of every value of the list. If an index is given, it will return the z-score of the given index.
def standardize(inputList: list | tuple, index: int = None, sample: bool = False):
    inputMean = mean(inputList)
    inputStd = std(inputList, sample)

    if index == None:
        return[round(((x - inputMean) / inputStd), 2) for x in inputList]
    else:
       return((inputList[index] - inputMean) / inputStd)

# Input a list or tuple, and return a sample of size n with or without replacement.
def sample(inputList: list | tuple, n: int, replacement: bool = False):
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
def quickMedian(inputList: list | tuple):
    length = len(inputList)
    if length % 2 == 0:
        return((quickSelect(inputList, length // 2 - 1) + quickSelect(inputList, length // 2)) / 2)
    
    else:
        return quickSelect(inputList, length // 2)
    
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
def Q1(inputList: list | tuple):
    sortedList = sorted(inputList)
    mid = len(sortedList) // 2

    # For an odd-length list, we exclude the middle element
    lower_half = sortedList[:mid] if len(sortedList) % 2 == 0 else sortedList[:mid]
    
    return median(lower_half)

# High Median: Median of the upper half of the sorted list
def Q3(inputList: list | tuple):
    sortedList = sorted(inputList)
    mid = len(sortedList) // 2

    # For an odd-length list, we exclude the middle element
    upper_half = sortedList[mid+1:] if len(sortedList) % 2 == 1 else sortedList[mid:]

    return median(upper_half)

def IQR(inputList: list | tuple):
    return Q3(inputList) - Q1(inputList)

class outlier:

    def sigma(inputList: list | tuple):
        mean = mean(inputList)
        std = std(inputList)
        return [x for x in inputList if not (mean - 2 * std <= x <= mean + 2 * std)]
                
    def IQR(inputList: list | tuple):
        median = median(inputList)
        IQR = IQR(inputList)
        return [x for x in inputList if not (median - 1.5 * IQR <= x <= median + 1.5 * IQR)]
                
    def test(inputList: list | tuple):
        outliers = outlier.sigma(inputList) + outlier.IQR(inputList)
        if outliers != []:
            return True
            
        else:
            return False

# Means and standard deviations of various distribution types.
class dstr:

    # A.K.A. z distribution
    class normal:

        def __init__(self, mean: float | int = 0, std: float | int = 1):
            self.mean = mean
            self.std = std
            self.var = std ** 2

        def __add__(dist1, dist2):
            meanNew = dist1.mean + dist2.mean
            stdNew = ((dist1.std ** 2) + (dist2.std ** 2)) ** 0.5
            return dstr.normal(meanNew, stdNew)
        
        def __sub__(dist1, dist2):
            meanNew = dist1.mean - dist2.mean
            stdNew = ((dist1.std ** 2) + (dist2.std ** 2)) ** 0.5
            return dstr.normal(meanNew, stdNew)
        
        
        def pdf(self, x: float | int):
            return (1 / (2 * pi * self.std ** 2) ** 0.5) * exp(-((x - self.mean) ** 2) / (2 * self.std ** 2))


        def cdf(self, a: float | int, b: float | int):
            phi = lambda x: 0.5 * (1 + erf((x - self.mean) / (self.std * (2) ** 0.5)))
            return phi(b) - phi(a)


    class t:
        def __init__(self, df: float | int):
            self.df = df

    class chisqr:
        def __init__(self, obs: list, exp: list = None):
            self.obs = obs

            # Check if obs is a matrix (list of lists)
            if all(isinstance(row, list) for row in self.obs):
                row_total = [sum(row) for row in self.obs]
                column_total = [sum(row[i] for row in self.obs) for i in range(len(self.obs[0]))]
                table_total = sum(row_total)

                self.exp = [[(row_total[i] * column_total[j]) / table_total for j in range(len(column_total))]
                            for i in range(len(row_total))]
                self.df = (row_total - 1) * (column_total - 1)
                self.stat = [[((obs - exp) ** 2) / exp for obs, exp in zip(obs_row, exp_row)] for obs_row, exp_row in zip(self.obs, self.exp)]

            elif exp is not None:
                # Check if exp is a matrix while obs is not, or vice versa
                if all(isinstance(row, list) for row in exp):
                    raise TypeError("The expected and observed values must either both be matrices or both be lists")
                else:
                    self.exp = exp
                    self.df = len(exp) - 1
                    self.stat = [((o - e) ** 2) / e for o, e in zip(self.obs, self.exp)]

            else:
                raise ValueError("Expected values must be provided for 1D observed data.")
                
    class sampling:
        
        # The sampling distribution of the sample proportion
        class prop:

            @validateProbability
            def __init__(self, p: float = None, n: int = None, *, mean: float = None, std: float = None, norm: bool = None):
                if p is not None and n is not None:
                    self.p = p
                    self.n = n
                    self.mean = p
                    self.std = ((p * (1 - p)) / n) ** 0.5
                    self.variance = self.std ** 2
                    if ((p * n > 10) or ((1 - p) * n > 10)) and norm == False:
                        self.norm = False
                    else:
                        self.norm = True
                elif mean is not None and std is not None:
                    self.p = mean
                    self.n = None
                    self.mean = mean
                    self.std = std
                    self.variance = std ** 2
                    self.norm = norm
                else:
                    raise ValueError("Provide either (p and n) or (mean, std, and normality)")

            def __sub__(dist1, dist2):
                new_mean = dist1.p - other.p
                new_std = ((dist1.p * (1 - dist1.p)) / dist1.n + (dist2.p * (1 - dist2.p)) / dist2.n) ** 0.5
                if (dist1.norm == True) and (dist2.norm == True):
                    new_norm = True
                else:
                    new_norm = False
                return dstr.sampling.prop(mean = new_mean, std = new_std, norm = new_norm)
            
            def pdf(self, x):
                if self.norm == True:
                    return dstr.normal(self.mean,self.std).pdf(x)
                else:
                    raise TypeError("The sampling distribution is not approximately normal")
                
            def cdf(self, x):
                if self.norm == True:
                    return dstr.normal(self.mean,self.std).cdf(x)
                else:
                    raise TypeError("The sampling distribution is not approximately normal")
                
        # The sampling distribution of sample mean
        class mean:
            
            def __init__(self, mean: float | int, std: float | int, n: int, norm : bool):
                self.mean = mean
                self.std = std / (n ** 0.5)
                self.n = n
                if (n >= 30) or norm == True:
                    self.norm = True
                else:
                    self.norm = False

            def __add__(dist1,dist2):
                new_mean = dist1.mean + dist2.mean
                new_std = (((dist1.std ** 2) / dist1.n) + (dist2.std ** 2) / dist2.n) ** 0.5
                if (dist1.norm == True) and (dist2.norm == True):
                    new_norm = True
                else:
                    new_norm = False
                return dstr.sampling.mean(new_mean,new_std,None,new_norm)
            
            def sub(dist1,dist2):
                new_mean = dist1.mean - dist2.mean
                new_std = (((dist1.std ** 2) / dist1.n) + (dist2.std ** 2) / dist2.n) ** 0.5
                if (dist1.norm == True) and (dist2.norm == True):
                    new_norm = True
                else:
                    new_norm = False
                return dstr.sampling.mean(new_mean,new_std,None,new_norm)
            
            def pdf(self, x):
                if self.norm == True:
                    return dstr.normal(self.mean,self.std).pdf(x)
                else:
                    raise TypeError("The sampling distribution is not approximately normal")
                
            def cdf(self, x):
                if self.norm == True:
                    return dstr.normal(self.mean,self.std).cdf(x)
                else:
                    raise TypeError("The sampling distribution is not approximately normal")
                
            
    # Describes the probability of x successes given n attempts and p probability of success.
    class binom:
        
        @validateProbability
        def __init__(self, p: probability, n: int):
            self.p = p
            self.n = n

        # The mean of the probability distribution, A.K.A. the expected amount of successes.
        def mean(p: probability, n: int):
            return(n * p)
            
        # The standard deviation of the probability distribution.
        def std(p: probability, n: int):
            return((n * p * (1 - p)) ** 0.5)
    
    # Describes the probability of taking x amount of attempts to get a success for p probability of success.
    class geomet:
        
        @validateProbability
        def __init__(self, p: probability):
            self.p = p

        # The expected amount of attempts for a success.
        def mean(p: probability):
            if p == 0:
                raise ZeroDivisionError("For geometric distributions, p cannot equal zero. Success can never be reached.")
            else:
                return(1/p)
        
        # The standard deviation of the probability distribution.
        def std(p: probability):
            if p == 0:
                raise ZeroDivisionError("For geometric distributions, p cannot equal zero. Success can never be reached.")
            else:
                return(((1-p) ** 0.5) / p)