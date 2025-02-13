Welcome to parametric.

This is my first project. I plan on making it continuous. Please do not hesitate to make suggestions to my code.
I wrote the first 4 functions on the same day I learned to code in Python. Expect some errors.

More detailed descriptions will be found wherever I put the documentation later.

Contents:

    Random Sample related functions:

        sample(inputList, n, replacement)
            Returns a random sample from inputList of size n.
            Samples with replacement if replacement == True, otherwise samples without replacement.

    Mean related functions:

    These relate to mean and standard deviation. Very useful and common in statistics.
    This will not contain the functions for a the mean/SD of binomial, geometric, etc. distributions.

        mean(inputList) | O(1)
            Takes a list of numbers, returns a mean. Simple enough.
            The mean is the average value of a set of numbers.
        
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

            zScorify(inputList, index = none, isSample)
                Turns every number in a set into a z-score.
                If an index is given, it will turn the number at the index into a z-score.
