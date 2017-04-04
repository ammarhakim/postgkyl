#!/usr/bin/env python
"""
Postgkyl module for field-particle correlations
"""
import numpy
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys


# --------------------------------------------------------------------
# Growth rate fitting stuff ------------------------------------------
def exp2(x, a, b):
    """Define custom exponential a*exp(2b*x)

    Parameters:
    x -- independent variable
    a -- scaling parameter
    b -- growth rate

    Notes:
    Energy (quantity^2) is often used for the growth-rate study,
    therefore the factor 2
    """
    return a*numpy.exp(2*b*x)


def fitGrowth(x, y, function=exp2, minN=100, maxN=None, p0=(1, 0.1)):
    """Fit function to continuously increasing region of data

    Parameters:
    x -- independet variable
    y -- dependent variable
    minN -- minimal number of fitted points (default: 100)
    maxN -- maximal number of fitted points (default: full length)
    function -- function to fit (default: exp2)
    p0 -- initial guess (default: 1, 0.1)

    Notes:
    The best is determined based on the coeficient of determination,
      R^2 https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    bestR2 = 0
    if maxN is None:
        maxN = len(x)
    bestParams = p0

    print('fitGrowth: fitting region {:d} -> {:d}'.format(minN, maxN))
    for n in numpy.linspace(minN, maxN-1, maxN-minN):
        n = int(n)
        xn = x[0 : n]  # continuously increasing fitting region
        yn = y[0 : n]
        try:
            params, cov = opt.curve_fit(function, xn, yn, bestParams)
            residual = yn - function(xn, *params)
            ssRes = numpy.sum(residual**2)
            ssTot = numpy.sum((yn - numpy.mean(yn))**2)
            R2 = 1 - ssRes/ssTot
            if R2 > bestR2:
                bestR2 = R2
                bestParams = params
                bestN = n
            percent = float(n-minN)/(maxN-minN)*100
            progress = '[' + int(percent/10)*'=' + (10-int(percent/10))*' ' + ']'
            sys.stdout.write(
                '\rgamma = {:+5.3f} (best {:+5.3f}) R^2 = {:6.4f}   {:6.2f}% done {}'.format(params[1], bestParams[1], R2, percent, progress))
            sys.stdout.flush()
        except RuntimeError:
            print('fitGrowth: curve_fit failed for N = {}'.format(n))

    print('\rgamma = {:+5.3f} (best {:+5.3f}) R^2 = {:6.4f}   {:6.2f}% done {}'.
          format(params[1], bestParams[1], R2, 100, '[==========]'))
    return bestParams, bestR2, bestN
