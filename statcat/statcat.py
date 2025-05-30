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

# General cdf function using trapezoidal method
def cdf(u, function, lower = -1, upper = 1, steps = 1000):
    if u <= lower:
        return 0
    elif u >= upper:
        return 1
    
    tot = 0.0
    
    step_size = (upper - lower) / steps

    x_prev = lower
    k_prev = function(x_prev)

    for i in range(1, steps + 1):
        x_curr = lower + i * step_size
        k_curr = function(x_curr)

        area = (k_prev + k_curr) * step_size / 2
        tot += area

        x_prev = x_curr
        k_prev = k_curr
    
    return tot

# Creates a list of ordered pairs using the cdf function
def precomp_cdf(function, cdf : bool = True, lower = -1, upper = 1, steps = 1000):
    u_vals = [lower + i * ((upper - lower) / steps) for i in range(steps + 1)]
    if cdf == true:
        cdfvals = [function(u) for u in u_vals]
    else:
        cdfvals = [cdf(u, function, lower, upper, steps) for u in u_vals]
    return list.zip(u_vals, cdfvals)

# The inverse of the cdf function
def invcdf(p, function, cdf : bool = True):
    u_vals = [lower + i * ((upper - lower) / steps) for i in range(steps + 1)]
    if cdf == True:
        cdfvals = [function(u) for u in u_vals]
    else:
        cdfvals = [cdf(u, function, lower, upper, steps) for u in u_vals]
    
    x, y = u_vals, cdf_vals
    if p <= 0:
        return x[0]
    elif p >= 1:
        return x[-1]
    
    low, high = 0, len(y) - 1
    while low < high:
        mid = (low + high) // 2
        if y[mid] < p:
            low = mid + 1
        else:
            high = mid

    i = max(1, low)

    x0, x1 = x[i-1], x[i]
    y0, y1 = y[i-1], y[i]
    t = (p - y0) / (y1 - y0)
    
    return x0 + t * (x1 - x0)

# Returns random noise based on a function
def noise(function, cdf : bool = True, n : Int = 1):
    return [invcdf(Random.uniform(0,1), function, cdf) for n in n

# Returns the rising factorial
def pochhammer(q, n):
    return prod(q + _ for _ in range(n))

# Returns the hypergeometric series with n steps. more n increases accuracy, but also increases computation time
def hypergeometric(a, b, c, z, n = 100):
    return sum(((pochhammer(a, i) * pochhammer(b, i)) / pochhammer(c, i)) * ((z ** i) / factorial(i)) for i in range(n))

# Beta function
def beta(z1, z2):
    return (gamma(z1) * gamma(z2)) / gamma(z1 + z2)

# Incomplete beta function
def inc_beta(x, a, b):
    return cdf(x, lambda t: (t ** (a - 1)) * ((1 - t) ** (b - 1)), 0, x)

# Refularized beta function
def reg_beta(x, a, b):
    return inc_beta(x, a, b) / beta(a, b)

# Incomplete gamma function
def inc_gamma(s, x):
    return cdf(x, lambda t: (t ** (s - 1)) * exp(-t), 0, x)

def kronkecker_delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def binomcoeff(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

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
        if total_weight == 0:
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
    
# Returns the standard deviation of the list.
def std(inputList, isSample : bool = False):
        return(variance(inputList, isSample) ** 0.5)

# Returns the z-score of a number.
def zScore(value, mean, std):
    return((value - mean) / std)

# Returns a list of the z-scores of every value of the list. If an index is given, it will return the z-score of the given index.
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
        tempList = inputList[:]

        for i in range(n): #Puts a random value in the set into the first n indices.
            j = randint(0,popSize - 1) # A random index j corresponding to the ordered index i. 
            tempList[i], tempList[j] = tempList[j], tempList[i] # Puts the jth element in the ith spot. Ends up with n random elements sampled without replacement.
        return(tempList[:n])

# Finds the most common value.
def mode(data):
    if not data:
        raise ValueError("mode() arg is an empty sequence")
    
    counts = {}
    for val in data:
        counts[val] = counts.get(val, 0) + 1
    
    max_count = -1
    mode_val = None
    for val, count in counts.items():
        if count > max_count:
            max_count = count
            mode_val = val
    return mode_val

# Finds a list of the most occurring values
def multimode(data):
    counts = {}
    for val in data:
        counts[val] = counts.get(val, 0) + 1
    
    max_count = max(counts.values())
    modes = [val for val, count in counts.items() if count == max_count]
    return sorted(modes)  # Optional: sorted for consistency

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
def quickMedian(inputList: list | tuple):
    length = len(inputList)
    if length % 2 == 0:
        return((quickSelect(inputList, length // 2 - 1) + quickSelect(inputList, length // 2)) / 2)
    
    else:
        return quickSelect(inputList, length // 2)
    
# If i'm being honest, I don't really know what this is supposed to do.
# But it returns the median of some frequencies and intervals and stuff.
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

def Q1(inputList: list | tuple):
    sortedList = sorted(inputList)
    mid = len(sortedList) // 2

    # For an odd-length list, we exclude the middle element
    lower_half = sortedList[:mid]
    
    return median(lower_half)

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
        return sorted((outlier.IQR(inputList)) + (outlier.sigma(inputList)))

class dstr:

    class arcsine:

        def __init__(self):
            self.mean = 1/2
            self.var = 1/3
            self.std = 1/3 ** 0.5
        
        def kernel(self, x):
            return 1 / ((x * (1 - x)) ** 0.5)

        def pdf(self, x):
            return (1 / pi) * self.kernel(x)
        
        def cdf(self, x):
            return (2 / pi) * asin(x ** 0.5)

    class beta:

            def __init__(self, alpha, beta):
                self.alpha = alpha
                self.beta = beta
                self.mean = alpha / (alpha + beta)
                self.var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
                self.std = self.var ** 0.5
            
            def kernel(self, x):
                return (x ** (self.alpha - 1)) * ((1 - x) ** (self.beta - 1))
            
            def pdf(self, x):
                return (1 / beta(self.alpha, self.beta)) * self.kernel(x)
            
            def cdf(self, x):
                return inc_beta(x, self.alpha, self.beta)

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

    class cauchy:

        def __init__(self, xnull, gamma):
            self.xnull = xnull
            self.gamma = gamma
            
        def kernel(self, x):
            return 1 / (1 + ((x - self.xnull) / self.gamma) ** 2)

        def pdf(self, x):
            return (1 / (pi * self.gamma)) * self.kernel(x)
        
        def cdf(self, x):
            return (1 / pi * atan(((x - self.xnull) / self.gamma))) + 1 / 2
        
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
                self.df = (len(row_total) - 1) * (len(column_total) - 1)
                self.cntrb = [[round(((obs - exp) ** 2) / exp, 5) for obs, exp in zip(obs_row, exp_row)] for obs_row, exp_row in zip(self.obs, self.exp)]
                self.stat = sum([sum(row) for row in self.cntrb])

            elif exp is not None:
                # Check if exp is a matrix while obs is not, or vice versa
                if all(isinstance(row, list) for row in exp):
                    raise TypeError("The expected and observed values must either both be matrices or both be lists")
                else:
                    self.exp = exp
                    self.df = len(exp) - 1
                    self.cntrb = [((o - e) ** 2) / e for o, e in zip(self.obs, self.exp)]
                    self.stat = sum(self.cntrb)

            else:
                raise ValueError("Expected values must be provided for 1D observed data.")
        
    class dirichlet:

        def __init__(self, alpha: list):
            self.K = len(alpha)
            self.alpha = alpha
            self.a0 = sum(alpha)
            self.atilde = [a_i / self.a0 for a_i in alpha]
            self.mean = [ai / self.a0 for ai in alpha]
            self.var = [(ai * (1 - ai)) / (self.a0 + 1) for ai in self.atilde]
            self.std = [v ** 0.5 for v in self.var]

        def kernel(self, x):
            return prod(xi ** (ai - 1) for xi, ai in zip(x, self.alpha))
        
        def pdf(self, x):
            return self.kernel(x) / prod(gamma(ai) for ai in self.alpha) / gamma(self.a0)

    class F:

        def __init__(self, d1, d2):
            self.d1 = d1
            self.d2 = d2
            if d2 > 2:
                self.mean = d2 / (d2 - 2)
            if d2 > 4:
                self.var = ((2 * d2 ** 2) * (d1 + d2 - 2)) / ((d1 * (d2 - 2) ** 2) * (d2 - 4))

        def pdf(self, x):
            return sqrt(pow(self.d1 * x, self.d1) * pow(self.d2, self.d2) / pow(self.d1 * x + self.d2, self.d1 + self.d2)) / (x * beta(self.d1 / 2, self.d2 / 2))

        def cdf(self, x):
            return reg_beta((self.d1 * x) / (self.d1 * x + self.d2), self.d1 / 2, self.d2 / 2)

    class gamma:

        def __init__(self, alpha, theta, *, rate):
            self.alpha = alpha
            if rate == None:
                self.theta = theta
                self.rate = 1 / theta
            elif rate or theta != None:
                raise AttributeError("Input either rate or scale")
            else:
                self.rate = rate
                self.theta = 1 / rate
            self.mean = alpha * theta
            self.var = alpha * (theta ** 2)
            self.std = self.var ** 0.5

        def kernel(self, x):
            return (x ** (self.alpha - 1)) * exp(-x * self.rate)
        
        def pdf(self, x):
            return (1 / (gamma(self.alpha)) * (self.theta ** self.alpha)) * self.kernel(x)
        
        def cdf(self, x):
            return (inc_gamma(self.alpha, x / self.theta)) / gamma(self.alpha)
        

    # Describes the probability of taking x amount of attempts to get a success for p probability of success.
    class geomet:
        
        @validateProbability
        def __init__(self, p: probability):
            self.p = p

        # The expected amount of attempts for a success.
        def mean(p: probability):
            return(1/p)
        
        # The standard deviation of the probability distribution.
        def std(p: probability):
            return(((1-p) ** 0.5) / p)
        
    class irwin_hall:
        
        def __init__(self, n):
            self.n = n
            self.mean = n / 2
            self.var = n / 12
            self.std = self.var ** 0.5

        def kernel(self, x):
            return sum(pow(-1, k) * binomcoeff(self.n, k) * pow(x - k, self.n - 1) for k in floor(x))
        
        def pdf(self, x):
            return (1 / (factorial(self.n - 1))) * self.kernel(x)
        
        def cdf(self, x):
            return (1 / factorial(self.n)) * sum(pow(-1, k) * binomcoeff(self.n, k) * pow(x - k, self.n) for k in floor(x))

    class lognorm:

        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma
            self.mean = exp(mu + (sigma ** 2) / 2)
            self.var = (exp(sigma ** 2) - 1) * exp(2* mu + sigma ** 2)
            self.std = self.var ** 0.5
        
        def kernel(self, x):
            return exp(-((log(x) - self.mu) ** 2) / (2 * self.sigma ** 2))
        
        def pdf(self, x):
            return (1 / (x * self.sigma * ((2 * pi) ** 0.5))) * self.kernel(x)
        
        def cdf(self, x):
            phi = lambda x: 0.5 * (1 + erf((x - self.mean) / (self.std * (2) ** 0.5)))
            return phi((log(x) - self.mu) / self.sigma)

    class maxboltz:
        
        def __init__(self, a):
            self.a = a
            self.mean = 2 * a * ((2 / pi) ** 0.5)
            self.var = ((a ** 2) * (3 * pi - 8)) / pi
            self.std = self.var ** 0.5
        
        def kernel(self, x):
            return ((x ** 2) / (self.a ** 3)) * exp((-x ** 2) / (2 * (self.a ** 2)))
        
        def pdf(self, x):
            return ((2 / pi) ** 0.5) * self.kernel(x)
        
        def cdf(self, x):
            return erf(x / (self.a * (2 ** 0.5))) - (((2 / pi) ** 0.5) * (x / self.a) * exp((-x ** 2) / (2 * (self.a ** 3))))

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
        
        def kernel(self, x:float | int):
            return exp(-((x - self.mean) ** 2) / (2 * self.std ** 2))
        
        def pdf(self, x: float | int):
            return (1 / (2 * pi * self.std ** 2) ** 0.5) * self.kernel(x)
        
        def cdf(self, a: float | int, b: float | int):
            phi = lambda x: 0.5 * (1 + erf((x - self.mean) / (self.std * (2) ** 0.5)))
            return phi(b) - phi(a)

    class pareto:

        def __init__(self, xm, alpha):
            self.xm = xm
            self.alpha = alpha
            if alpha <= 1:
                self.mean = inf
            else:
                self.mean = (alpha * xm) / (alpha - 1)
            
            if alpha <= 2:
                self.var = inf
                self.std = inf
            else:
                self.var = (xm ** 2) / (((alpha - 1) ** 2) * (alpha - 2))
                self.std = self.var ** 0.5
        
        def pdf(self, x):
            return (self.alpha * self.xm ** self.alpha) / (x ** (self.alpha + 1))
        
        def cdf(self, x):
            return 1 - ((self.xm / x) ** self.alpha)

    class t:
        def __init__(self, df: float | int):
            self.df = df
            self.mean = 0
            self.var = df / (df - 2)
            self.std = self.variance ** 0.5
        
        def kernel(self, x):
            return (1 + ((x ** 2) / self.df)) ** (-(self.df + 1) / 2)
        
        def pdf(self, x):
            return gamma((self.df + 1) / 2) / (((self.df ** pi) ** 0.5) * gamma(self.df/2))
        
        def cdf(self, x, steps):
            return 1/2 + (x * gamma((self.df + 1) / 2)) * ((hypergeometric(1/2, (self.df + 1) / 2, 3/2, -(x ** 2 / self.df))) / (((self.df ** pi) ** 0.5) * gamma(self.df/2)))

    # I was gonna put the Wishart distribution here but it frightens me
          
    class sampling:
        
        # The sampling distribution of the sample proportion
        class prop:

            @validateProbability
            def __init__(self, p: float = None, n: int = None, *, mean: float = None, std: float = None, norm: bool = False):
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
                new_mean = dist1.p - dist2.p
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
            
            def __sub__(dist1,dist2):
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
                
        # This is black magic to me.
        # I'm gonna try my best.
        class kde:

            def __init__(self, data, h = 'scott', kernel = None ): # None turns into dstr.normal().kernel at runtime
                self.data = data
                self.ndata = len(self.data)
                if isinstance(h, str):
                    if h == 'scott':
                        std = std(data)
                        self.h = std * self.n ** (-1/5)
                    elif h == 'silverman':
                        std = std(data)
                        iqr = outlier.iqr(data)
                        self.h = 0.9 * min(std, iqr / 1.34) * self.n ** (-1/5)
                    else:
                        raise ValueError(f"Unknown bandwidth rule: {h}")
                else:
                    self.h = h
                if kernel is None:
                    kernel = dstr.normal().kernel
                self.kernel = kernel
                
            def pdf(self, x):
                return (
                    1/(self.n * self.h) * sum(self.kernel((x - x_i) / self.h) for x_i in self.data)
                )
            
            def mode(self):
                data_min, data_max = min(self.data), max(self.data)
                margin = self.h * 3
                start = data_min - margin
                end = data_max + margin
                
                steps = 1000
                step_size = (end - start) / steps
                
                max_x = start
                max_pdf = self.pdf(start)
                
                for i in range(1, steps + 1):
                    x = start + i * step_size
                    p = self.pdf(x)
                    if p > max_pdf:
                        max_pdf = p
                        max_x = x
                
                return max_x
            
            def random(self, n = 1):
                bandwidth = self.h * std(self.data)
                samples = []

                for _ in range(n):
                    xi = sample(self.data, 1)
                    u = noise()
                    samples.append(xi + bandwidth * u)

                return samples
            
# kernels are sick
class kernel:

    @staticmethod
    def uniform(u):
        if -1 <= u <= 1:
            return 1/2
        else:
            return 0
        
    @staticmethod
    def triweight(u):
        if -1 <= u <= 1:
            return 35/32 * (1 - (abs(u) ** 2) ** 2)
        else:
            return 0
        
    @staticmethod
    def tricube(u):
        if -1 <= u <= 1:
            return 35/32 * (1 - (abs(u) ** 3) ** 3)
        else:
            return 0
        
    @staticmethod
    def triangular(u):
        if -1 <= u <= 1:
            return 1 - abs(u)
        else:
            return 0
        
    @staticmethod
    def silverman(u):
        return 1/2 * exp(-(abs(u) / (2 ** 0.5))) * sin((abs(u) / (2 ** 0.5)) + pi / 4)
    
    @staticmethod
    def sigmoid(u):
        return (2 / pi) * (1 / (exp(u) + exp(-u)))
    
    @staticmethod
    def biweight(u):
        if -1 <= u <= 1:
            return 15/16 * (1 - (u ** 2) ** 2)
        else:
            return 0
        
    @staticmethod
    def logistic(u):
        return 1 / (exp(u) + 2 + exp(-u))
    
    @staticmethod
    def parabolic(u):
        if -1 <= u <= 1:
            return 3/4 * (1 - (u ** 2))
        else:
            return 0
        
    @staticmethod
    def cosine(u):
        if -1 <= u <= 1:
            return pi/4 * cos((pi/2) * u)
        else:
            return 0

# Regression
class reg:
    
    # Linear
    class lin:

        def __init__(self, xdata, ydata, *, data):
            if data != None:
                self.data = data
                self.xdata, self.ydata = zip(*data)
            else:
                self.xdata = xdata
                self.ydata = ydata
                self.data = [(x,y) for x, y in zip(xdata,ydata)] 
            self._compute_stats()

        def _compute_stats(self):
            self.x_bar = mean(self.xdata)
            self.y_bar = mean(self.ydata)
            self.xdev = [x - self.x_bar for x in self.xdata]
            self.ydev = [y - self.y_bar for y in self.ydata]
            self.b = (sum([(x * y) for x, y in zip (self.xdata,self.ydata)])) / sum([(x ** 2) for x in self.xdev])
            self.a = self.y_bar - (self.b * self.x_bar)
            self.residuals = [(y - (self.a + self.b * x)) for x, y in zip(self.xdata,self.ydata)] 
            self.rsq = 1 - (sum([r ** 2 for r in self.residuals]) / sum([y ** 2 for y in self.ydev]))
            self.r = self.rsq ** 0.5
            self.s = sum([y ** 2 for y in self.ydev]) / (len(self.data) - 2)

        def eq(self, x):[
            return self.a + self.b * x

