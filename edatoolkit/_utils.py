import math
import numpy as np
from copy import copy, deepcopy
import sys
# testBit() returns a nonzero result, 2**offset, if the bit at 'offset' is one.

def progressBar(loc):
    sys.stdout.flush()
    bar = "\r" + "="*int(round(loc*60)) + ">" + " "*(60 - int(round(loc*60))) + "|%d%%"%int(round(loc*100))
    sys.stdout.write(bar)


def normaliseVector(values, timeResolution=100, method='linear'):
    """Reshape a vector to a given resolution (number of samples).
    
    This method gracefully handles Nan values by propogating them to the
    normalised version of the vector. Interpolation based methods will
    propogate Nan if they are needed for interpolation.
    Aggregation based methods will ignore Nans.
       
    ----Parameters----
    values  -- The vector to be normalised (reshaped)
    timeResolution  -- (optional) New number of samples (default 100)
    method -- (optional) Method used to map values. Default 'linear'
               Possible values are:
               'linear' - use linear interpolation between adjacent points
               'cubic' - use bi-cubic interpolation between adjacent points
               'mean' - aggregation method which uses the mean of all values
                        in a given bin (timeResolution < len(values))
               'max' - use the max value for values in that bin (timeResolution < len(values))
               
    ----Returns----
    A vector that is resampled to a specific length.
                        
    ----Exceptions----
    IOError - if a scalar or empty matrix is given
    """  
    import numpy
    from scipy.interpolate import interp1d
    import scipy.stats
    
    timeResolution = int(timeResolution) 
    nValues = values.size
    if nValues < 2:
        raise IOError('Input must be a vector (at least 2 values)')
    if method == 'linear':
        newX = numpy.linspace(0, 1, timeResolution)
        oldX = numpy.arange(0, nValues, dtype='float64') / (nValues - 1)
        normValues = numpy.interp(newX, oldX, values);
    elif method == 'cubic':
        newX = numpy.linspace(0, 1, timeResolution)
        oldX = numpy.arange(0, nValues, dtype='float64') / (nValues - 1)
        interpolator = scipy.interpolate.interp1d(oldX, values, kind='cubic')
        normValues = interpolator(newX)
    elif method == 'mean':
        normValues = numpy.zeros((timeResolution), 'float64')
        bounds = numpy.linspace(0, nValues, timeResolution + 1)
        for index, (lbound, rbound) in enumerate(zip(bounds[:-1], bounds[1:])):
            normValues[index] = scipy.stats.nanmean(values[lbound:rbound])
    elif method == 'max':
        normValues = numpy.zeros((timeResolution), 'float64')
        bounds = numpy.linspace(0, nValues, timeResolution + 1)
        for index, (lbound, rbound) in enumerate(zip(bounds[:-1], bounds[1:])):
            validValues = values[lbound:rbound]
            validIndex = ~numpy.isnan(validValues)
            if numpy.sum(validIndex) > 0:
                normValues[index] = numpy.max(validValues[validIndex])
            else:
                normValues[index] = numpy.nan
    else:
        raise ValueError('Unrecognized method : ' + method)
    return normValues



def testBit(int_type, offset):
    mask = 1 << offset
    return(int_type & mask)

# setBit() returns an integer with the bit at 'offset' set to 1.

def setBit(int_type, offset):
    mask = 1 << offset
    return(int_type | mask)

# clearBit() returns an integer with the bit at 'offset' cleared.

def clearBit(int_type, offset):
    mask = ~(1 << offset)
    return(int_type & mask)

# toggleBit() returns an integer with the bit at 'offset' inverted, 0 -> 1 and 1 -> 0.

def toggleBit(int_type, offset):
    mask = 1 << offset
    return(int_type ^ mask)

def ByteToHex( byteStr ):
    """
    Convert a byte string to it's hex string representation e.g. for output.
    """
    
    # Uses list comprehension which is a fractionally faster implementation than
    # the alternative, more readable, implementation below
    #   
    #    hex = []
    #    for aChar in byteStr:
    #        hex.append( "%02X " % ord( aChar ) )
    #
    #    return ''.join( hex ).strip()        

    return ''.join( [ "%02X " % ord( x ) for x in byteStr ] ).strip()

#-------------------------------------------------------------------------------

def HexToByte( hexStr ):
    """
    Convert a string hex byte values into a byte string. The Hex Byte values may
    or may not be space separated.
    """
    # The list comprehension implementation is fractionally slower in this case    
    #
    #    hexStr = ''.join( hexStr.split(" ") )
    #    return ''.join( ["%c" % chr( int ( hexStr[i:i+2],16 ) ) \
    #                                   for i in range(0, len( hexStr ), 2) ] )
 
    bytes = []

    hexStr = ''.join( hexStr.split(" ") )

    for i in range(0, len(hexStr), 2):
        bytes.append( chr( int (hexStr[i:i+2], 16 ) ) )

    return ''.join( bytes )

def int2bin(i, n):
    """Convert decimal integer i to n-bit binary number (string).

    >>> int2bin(0, 8)
    '00000000'

    >>> int2bin(123, 8)
    '01111011'

    >>> int2bin(123L, 8)
    '01111011'

    >>> int2bin(15, 2)
    Traceback (most recent call last):
    ValueError: Value too large for given number of bits.

    """
    hex2bin = {'0': '0000', '1': '0001', '2': '0010', '3': '0011',
               '4': '0100', '5': '0101', '6': '0110', '7': '0111',
               '8': '1000', '9': '1001', 'a': '1010', 'b': '1011',
               'c': '1100', 'd': '1101', 'e': '1110', 'f': '1111'}
    # Convert to hex then map each hex digit to binary equivalent.
    result = ''.join([hex2bin[x] for x in hex(i).lower().replace('l','')[2:]])
                      
    # Shrink result to appropriate length.
    # Raise an error if the value is changed by the truncation.
    if '1' in result[:-n]:
        raise ValueError("Value too large for given number of bits.")
    result = result[-n:]
    # Zero-pad if length longer than mapped result.
    result = '0'*(n-len(result)) + result
    return result


def bin2int(bin_string):
    """Convert binary number string to decimal integer.
    
    Note: Python > v2 has int(bin_string, 2)

    >>> bin2int('1111')
    15

    >>> bin2int('0101')
    5

    """
##     result = 0
##     bin_list = list(bin_string)
##     if len(filter(lambda x: x in ('1','0'), bin_list)) < len(bin_list):
##         raise Exception ("bin2int: Error - not a binary number: %s"
##                          % bin_string)
##     bit_list = map(int, bin_list)
##     bit_list.reverse()  # Make most significant bit have highest index.
##     for bit_place in range(len(bit_list)):
##         result = result + ((2**bit_place) * bit_list[bit_place])
##     return result
    return int(bin_string, 2)


def reverse(input_string):
    """Reverse a string. Useful for strings of binary numbers.

    >>> reverse('abc')
    'cba'

    """
    str_list = list(input_string)
    str_list.reverse()
    return ''.join(str_list)

def flip(b, bits=16):
	return bin2int(reverse(int2bin(b,n)))

def truncate(number, points=0):
	return math.trunc(number*pow(10, points))/float(pow(10, points))


def isOverlapInLogs(logfiles):
	timespans = []
	for log in logfiles:
		span = np.arange(time.mktime(log.startTime.timetuple()), time.mktime(log.endTime.timetuple()))
		timespans.append(span)
	isOverlap = len(set.intersection(*map(set,timespans))) > 0
	if isOverlap:
		return True
	else:
		return False

def movingAverageFilter(signal, window_size=2):  
	WINDOW = window_size
	data = deepcopy(signal)
	extended_data = np.hstack([[data[0]] * (WINDOW- 1), data])
	weightings = np.repeat(1.0, WINDOW) / WINDOW
	return np.convolve(extended_data, weightings)[WINDOW-1:-(WINDOW-1)]

