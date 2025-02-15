Welcome to rhoPy.

This is my first project. I plan on making it continuous. Please do not hesitate to make suggestions to my code.
I wrote the first 4 functions on the same day I learned to code in Python. Expect some errors.

More detailed descriptions will be found wherever I put the documentation later.

WHAT I DON'T HAVE THAT statistics DOES:
    median
    median_low
    median_high
    median_grouped
    mode
    multimode
    fmean
    quantiles
    kde
    kde_random
    covariance
    correlation
    linear_regression

    normal distribution object


WHAT I HAVE THAT statistics DOESN'T:
    weighted mean
    weighted geometric mean
    weighted harmonic mean
    z-scorify
    z-score (for a set of inputs)
    sample (for a set of inputs)
    sampling distribution calculations
    binomial distribution calculations
    geometric distribution calculations
   


Contents:

    Random Sample related functions:

        sample(inputList, n, replacement)
            Returns a random sample from inputList of size n.
            Samples with replacement if replacement == True, otherwise samples without replacement.

    Mean related functions:

    These relate to mean and standard deviation. Very useful and common in statistics.
    This will not contain the functions for a the mean/SD of binomial, geometric, etc. distributions.

        mean(inputList) | O(1) with weights: O(n)
            Takes a list of numbers, returns the arithmetic mean. Simple enough.
            The mean is the average value of a set of numbers.
            If a list of weights is input, it will return the weighted mean of the numbers.
                Author's note: surprisingly useful! Average atomic mass, discrete probabilities, etc.

        geometMean | O(1), with weights O(n)
            Takes a list, returns the geometric mean.
            If a list of weights is input, returns the weighted geometric mean.
        
        harmMean | O(1), with weights O(n)
            Don't worry. It won't hurt you.
            Returns the harmonic mean of a set of numbers.
            Guess what's gonna get returned if you give it a list of weights.
        
        variance(inputList, isSample) | O(n)
            Takes a list of numbers, and returns the variance.
            If isSample is true, it returns the sample variance. Otherwise, it returns the population variance.
            Variance is the average squared distance of all numbers in a set from the mean.

        stDev(inputList, isSample) | O(n)
            Square roots the variance.
            Standard deviation is the average distance of all numbers in a set from the mean.

        Z-score related functions:

            The z-score is the amount of standard deviations a value is from the mean. A unitless value.
            These functions can also be used for the standardized test statistic and standard error.

            zScore(inputValue, inputMean, stDev) | O(1)
                Takes the z-score of a single value given a mean and standard deviation.

            zScorify(inputList, index = none, isSample) | O(n)
                Turns every number in a set into a z-score.
                If an index is given, it will turn the number at the index into a z-score.

    Class dists
        Distribution related classes and functions.

            Class sampling
                Classes related to sampling distributions.

                    Class prop

                        The mean of the sampling distribution for p-hat (sample proportion) is the population proportion (rho).

                        stDev(p: probability, n: int)
                            Returns the standard deviation of the probability distribution.
                    
                    Class mean

                        The mean of the sampling distribution for  x-bar (sample mean) is the population mean (mu).

                        stDev(sigma: float | int, n: int)
                            Returns the standard deviation of the probability distribution.
                
            Class binom
                A distribution of the probability of getting x amount of successes for n attempts with p probability

                    mean(p: probability, n: int)
                        Returns the mean of the distribution.
                        A.K.A. the expected amount of successes.
                                                                                pssst: just multiply the two numbers. If you needed a library for just this, I pity you.
                    stDev(p: probability, n: int)
                        Returns the standard deviation of the distribution.

            Class geomet
                A distribution describing the amount of attempts required for x successes
                p cannot ever equal zero. (Unless you want a ZeroDivisionError)
                    mean(p: probability)
                        Returns the mean of the distribution.
                        The expected amount of attempts until one success.

                    stDev(p: probability)
                        Returns the standard deviation of the distribution.

                    



