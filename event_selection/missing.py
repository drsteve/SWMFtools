import random
import numpy as np
import spacepy.toolbox as tb
from scipy.ndimage.filters import gaussian_filter


def fill_gaps(data, fillval=9999999, sigma=5, winsor=0.05, noise=False, constrain=False,
              method='linear'):
    '''Fill gaps in input data series, using interpolation plus noise

    The noise approach is based on Owens et al. (Space Weather, 2014).

    data - input numpy ndarray-like
    fillval - value marking fill in the time series
    sigma - width of gaussian filter for finding fluctuation CDF
    winsor - winsorization threshold, values above p=1-winsor and below p=winsor are capped
    noise - Boolean, if True add noise to interpolated region, if False use interp only
    constrain - Boolean, if True
    method - string. Specifies interpolation type, options are 'linear' or 'sigmoid'
    '''
    # identify sequences of fill in data series
    gaps = np.zeros((len(data), 2), dtype=int)
    k = 0
    for i in range(1, len(data)-1):
        # Single space gap/fillval
        if (tb.feq(data[i], fillval)) and (~tb.feq(data[i+1], fillval)) and (~tb.feq(data[i-1], fillval)):
            gaps[k][0] = i
            gaps[k][1] = i
            k += 1
        # Start of multispace gap/fillval
        elif (tb.feq(data[i], fillval)) and (~tb.feq(data[i-1], fillval)):
            gaps[k][0] = i
        # End of multispace gap/fillval
        elif (tb.feq(data[i], fillval)) and (~tb.feq(data[i+1], fillval)):
            gaps[k][1] = i
            k += 1
    gaps = gaps[:k]

    # if no gaps detected
    if k == 0:
        return data

    # fill gaps with requested interpolation scheme
    interp_dict = {'linear': fill_linear,
                   'sigmoid': fill_sigmoid}
    if method not in interp_dict:
        raise ValueError('Requested interpolation method ({}) not supported'.format(method))
    for gap in gaps:
        data = interp_dict[method](data, gap)

    if noise:
        # generate CDF from delta var
        series = data.copy()
        smooth = gaussian_filter(series, sigma)
        dx = series-smooth
        dx.sort()
        p = np.linspace(0, 1, len(dx))
        # "Winsorize" - all delta-Var above/below threshold at capped at threshold
        dx[:p.searchsorted(0.+winsor)] = dx[p.searchsorted(0.+winsor)+1]
        dx[p.searchsorted(1.-winsor):] = dx[p.searchsorted(1.-winsor)-1]

        # draw fluctuations from CDF and apply to linearly filled gaps
        for gap in gaps:
            for i in range(gap[1]-gap[0]+1):
                series[gap[0]+i] += dx[p.searchsorted(random.random())]

        # cap variable if it should be strictly positive (e.g. number density)
        # use lowest measured value as floor
        if constrain and series.min() > 0.0:
            series[series < series.min()] = series.min()
        return series

    return data


def fill_linear(data, gap):
    """apply linear fill to region of input array

    data - input array
    gap  - indices marking region for fill
    """
    a = data[gap[0] - 1]
    b = data[gap[1] + 1]
    dx = (b - a)/(gap[1] - gap[0] + 2)
    for i in range(gap[1] - gap[0] + 1):
        data[gap[0] + i] = a + dx*(i + 1)
    return data


def fill_sigmoid(data, gap, tr=None, sl=None):
    """apply S-shaped (sigmoidal) fill to region of input array

    data - input array
    gap - indices marking region for fill
    tr - transition point (0, 1)
    sl - slope term (0, 1). 0 is equivalent to linear interpolation,
         as slope tends to 1 the function tends to a step.
    
    Modified from: reddit.com/r/gamedev/comments/4xkx71/sigmoidlike_interpolation/
    """
    a = data[gap[0] - 1]
    b = data[gap[1] + 1]
    def sigfunc(xval, trans=0.5, slope=0.5, up=True):
        cexp = 2/(1 - slope) - 1
        if up:
            if xval <= trans:
                numer = xval**cexp
                denom = trans**(cexp - 1)
                fval = numer/denom
            else:
                numer = (1 - xval)**cexp
                denom = (1 - trans)**(cexp - 1)
                fval = 1 - numer/denom
        else:
            if xval <= trans:
                numer = xval**cexp
                denom = 1 - trans**(cexp - 1)
                fval = 1 - numer/denom
            else:
                numer = (1 - xval)**cexp
                denom = (1 - trans)**(cexp - 1)
                fval = numer/denom
        return fval
    if a > b:
        # slope downward
        if sl is None:
            sl = 0.3
        if tr is None:
            tr = 0.35
        up = False
        hi, lo = a, b
    elif b > a:
        # slope upward
        if sl is None:
            sl = 0.7
        if tr is None:
            tr = 0.5
        up = True
        hi, lo = b, a
    else:
        data = fill_linear(data, gap)
        return data
    gaplen = gap[1] - gap[0] + 1
    for i in range(gaplen):
        fracx = i/gaplen
        data[gap[0] + i] = (hi - lo) * sigfunc(fracx, trans=tr, slope=sl, up=up) + lo
    return data
