import os
import sys
cmd_folder = os.path.dirname(os.path.abspath(__file__))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
lib_folder = os.path.abspath(os.path.dirname("../../../lib/python/"))
if lib_folder not in sys.path:
    sys.path.insert(0, lib_folder)
lib_folder = os.path.abspath(os.path.dirname("../../lib/python/"))
if lib_folder not in sys.path:
    sys.path.insert(0, lib_folder)

from qLogFile import *
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import scipy.stats
from matplotlib import pyplot as plt

import ais.logger
afflogger = ais.logger.get_affdex_logger()

class Interval:
    def __init__(self, start, end, sampleRate=8.0):
        self.start = start
        self.end = end
        self.sampleRate = sampleRate
        self.duration = (end - start)/self.sampleRate

def computePPMForSubset(log, increment=60, start=None, end=None):
    mylog = deepcopy(log)
    if start != None and end !=None:
        signal = removeBaseline(lowpass(medfilter(mylog), 3.0))[start : end]
    else:
        signal = removeBaseline(lowpass(medfilter(mylog), 3.0))
    ppm = []
    threshold = 0.018
    sr = int(log.sampleRate)
    spm = 60*sr
    step = sr*increment #Sliding window increments: 1 second
    maxCompleteIndex = (len(signal) - 1 - spm) - ((len(signal) - spm)%step)
    scrPeriod = []
    period = 0
    afflogger.info("Analyzing...")
    try:
        for n in range(1, maxCompleteIndex, step):
            scrsPerMinute = 0
            #print "Calculating PPM for " + str(n) + "    :    " + str(n+spm-1) +  "  |  " + str(n+spm-1 - n)
            for k in range(n, n + (spm - 1)):
                if signal[k] > threshold and signal[k-1] < threshold:
                    scrsPerMinute = scrsPerMinute + 1
            ppm.append(scrsPerMinute)#             
    except:
        afflogger.info("Problem computing PPM; results may not be valid")

    return ppm


def computePPM(log, increment=60, threshold=0.018):
    if not log.filename.startswith("np"):
        if len(log.data) < 10000:
            mylog = deepcopy(log)
        else:
            mylog = log
        afflogger.info("Filtering...")
        afflogger.info("Removing baseline")
        signal = removeBaseline(lowpass(medfilter(mylog), 3.0))
    else:
        afflogger.info("Not Filtering...")
        afflogger.info("Not Removing baseline")
        signal = deepcopy(log.EDA())
    ppm = []
    sr = int(log.sampleRate)
    spm = 60*sr
    step = sr*increment #Sliding window increments: 1 second
    maxCompleteIndex = (len(signal) - 1 - len(signal)%spm) - ((len(signal) - spm)%step)
    scrPeriod = []
    period = 0
    afflogger.info("Analyzing...")
    try:
        for n in range(1, maxCompleteIndex, step):
            scrsPerMinute = 0
            #print "Calculating PPM for " + str(n) + "    :    " + str(n+spm-1) +  "  |  " + str(n+spm-1 - n)
            for k in range(n, n + spm -1):
                if signal[k] > threshold and signal[k-1] < threshold:
                    scrsPerMinute = scrsPerMinute + 1
            ppm.append(scrsPerMinute)
    except:
        afflogger.warn("Problem computing PPM; results may not be valid")

    return ppm
    

def correlateAcc(log):
    log2 = log
    if log.startTime > log2.endTime or log2.startTime > log.endTime:
        afflogger.warn("No overlap; correlation not possible")
    else:
        signal1 = removeBaseline(log)
        signal2 = log2.ACC_X()
        st1 = 0
        stp1 = len(signal1) - 1
        st2 = 0
        stp2 = len(signal2) - 1
        if log.startTime < log2.startTime:
            st1 = int(log.offsetForTime(log2.startTime))
        elif log2.startTime < log.startTime:
            st2 = int(log2.offsetForTime(log.startTime))
        if log2.endTime > log.endTime:
            stp2 = int(log2.offsetForTime(log.endTime))
        elif log2.endTime < log.endTime:
            stp1 = int(log.offsetForTime(log2.endTime))
        afflogger.info("Signal 1: " + str(st1) + " | " + str(stp1))
        afflogger.info("Signal 2: " + str(st2) + " | " + str(stp2))
        signal1 = np.array(signal1[st1 : stp1])
        signal2 = np.array(signal2[st2 : stp2])
        if log.sampleRate > log2.sampleRate or len(signal1) > len(signal2):
            signal1 = scipy.signal.resample(signal1, len(signal2))
            sr = log2.sampleRate
        elif log2.sampleRate > log.sampleRate or len(signal2) > len(signal1):
            signal2 = scipy.signal.resample(signal2, len(signal1))
            sr = log.sampleRate
        else:
            sr = log.sampleRate
        afflogger.info("Signal1: " + str(len(signal1)))
        afflogger.info("Signal2: " + str(len(signal2)))
        correlation = scipy.stats.pearsonr(signal1, signal2)
        afflogger.info("r=" + str(correlation))
        corr = []
        spm = 60 * sr
        for n in range(0, len(signal1) - spm, spm):
            sub1 = signal1[n : n + spm]
            sub2 = signal2[n : n + spm]
            corr.append(scipy.stats.pearsonr(sub1, sub2)[0])
        plt.subplot(211)
        plt.plot(signal1, color = "blue")
        plt.plot(signal2, color = "red")
        plt.ylabel("EDA ($\mu$S)")
        plt.subplot(212)
        track = np.arange(0, len(corr), spm)
        plt.plot(corr)
        #plt.bar(track, corr, spm)
        plt.ylabel("Correlation")
        if len(sys.argv) > 4 and (sys.argv[4] == "save"):
            afflogger.info("Saving figure")
            title = log.filename.rstrip('.eda') + "_PPM"
            if os.path.isdir(sys.argv[1]):
                filename = os.path.join(sys.argv[1], title)
            else:
                filename = title
            plt.savefig(filename, dpi=200)
        else:
            afflogger.info("Showing figure")
            plt.show()
        plt.clf()



def removeOutliers(log):
    EDA = log.EDA()
    max = scipy.stats.scoreatpercentile(EDA, 90)
    min = scipy.stats.scoreatpercentile(EDA, 10)
    filteredSignal = clampSignal(EDA, min, max)
    plt.plot(EDA)
    plt.plot(filteredSignal)
    plt.show()



class accStats:
    def __init__(self, logfile):
        self.log = logfile
        self.x = np.array(self.log.ACC_X())
        self.y = np.array(self.log.ACC_Y())
        self.z = np.array(self.log.ACC_Z())
        self.sampleRate = self.log.sampleRate
    
    def jerk(self):
        jx = np.diff(x)
        jy = np.diff(y)
        jz = np.diff(z)
        return (jx, jy, jz)
    
    def work(self):
        return np.sqrt(self.x*self.x + self.y * self.y + self.z * self.z)
    
    #Below is shaving specific code----------------------------
    def getShaveIntervals(self, range=2, axisToUse="X", startUp=True):
        if axisToUse == "X":
            axis = self.x
        elif axisToUse == "Y":
            axis = self.y
        elif axisToUse == "Z":
            axis = self.z
        else:
            axis = self.y

        #firstSign = ((axis[0]>0)-0.5)*2 # yields either 1 or -1 depending on the sign of the first element in the z-axis accl signal.
        firstSign = 1 if startUp else -1
        #rinse_intervals = (self.z*firstSign) > 0    # yields a binary vector, 1 if element is whithin a rinse interval, 0 otherwise
        #stroke_intervals = (self.z*firstSign) < 0   # yields a binary vector, 1 if element is whithin a shaving interval, 0 otherwise
        rinse_intervals = (scipy.ndimage.filters.gaussian_filter1d(axis, range*2+1, order=0) * firstSign) > 0    # yields a binary vector, 1 if element is whithin a rinse interval, 0 otherwise
        stroke_intervals = (scipy.ndimage.filters.gaussian_filter1d(axis, range*2+1, order=0) * firstSign) < 0   # yields a binary vector, 1 if element is whithin a shaving interval, 0 otherwise

        change_rinse = np.diff(rinse_intervals*10)# rinse_intervals contains 1 or shaving intervals and zero otherwise, np.diff() yields 1 when the rinse intervals start and -1 when intervals end. Multiply by 10 to ensure array is interpreted as int, not boolean (OWS)
        change_stroke = np.diff(stroke_intervals*10)
        stroke_start = np.nonzero(change_stroke > 0)[0] + 1  # getting locations of when rinse intervals start (1-values obtained from np.diff()). +1 to get the index of first element in rinse interval and not the element preceeding it, np.diff() = forward np.difference
        stroke_end   = np.nonzero(change_stroke < 0)[0]         # -1 values indicate the end of rinse intervals as obtained through the np.diff(), getting the locations of when this occurs.
        rinse_start = np.nonzero(change_rinse < 0)[0] + 1
        rinse_end = np.nonzero(change_rinse > 0)[0]
        
        if(len(stroke_end) > len(stroke_start)): # the above does not see the end of the signal as the end of a rinse interval, we have to add the end explicitly.
            #stroke_end = stroke_end[0 : len(stroke_intervals)-1]; 
            stroke_end = [x for x in stroke_end if x > stroke_start[0]]
        elif(len(stroke_end)  < len(stroke_start)): # the above does not see the end of the signal as the end of a rinse interval, we have to add the end explicitly.
            stroke_start = stroke_start[:-1]; 

        strokePeriods = self.makeIntervals(stroke_start,stroke_end)
        #stroke_start
        #stroke_end
        rinsePeriods = self.makeIntervals(np.insert(stroke_end, 0, 0), np.append(stroke_start, len(axis)-1))
        #return (strokePeriods, rinsePeriods)
        return (rinsePeriods, strokePeriods)
    
    def makeIntervals(self, start_array, end_array):
        if len(start_array) != len(end_array):
            afflogger.info("Exception -- start and end arrays must be of same length")
            afflogger.info(start_array)
            afflogger.info(end_array)
            return False
        else:
            intervals = []
            for n in range(0, len(start_array)):
                intervals.append(Interval(start_array[n], end_array[n], sampleRate=self.sampleRate))
            return np.array(intervals)
    
    def findStrokes(self, strokeStart, strokeEnd, threshold_slope=0.65, smooth=2, window=2, axisToUse="Y"): #threshold_slope in change per second, not per sample
        if axisToUse == "X":
            axis = self.x[strokeStart : strokeEnd]
        elif axisToUse == "Y":
            axis = self.y[strokeStart : strokeEnd]
        elif axisToUse == "Z":
            axis = self.z[strokeStart : strokeEnd]
        else:
            axis = self.y[strokeStart : strokeEnd]
        if smooth > 1:
            daxis = np.insert(np.diff(scipy.ndimage.filters.gaussian_filter(axis, smooth)), 0, 0) #To keep vector length constant
        else:
            daxis = np.insert(np.diff(axis), 0, 0) #To keep vector length constant
        if window%2 == 0:
            window = window + 1
        maxis = scipy.ndimage.filters.maximum_filter1d(axis, window) == axis
        weight = 0.5
        threshold = threshold_slope/self.sampleRate
        zeroCrossings = []
        for n in range(0, len(axis)-1):
            if n == 0:
                preSlope =daxis[n]
            else:
                preSlope = daxis[n]*(1.0-weight) + daxis[n-1]*(weight)
            if n == len(axis)-2:
                postSlope = daxis[n+1]
            else:
                postSlope = daxis[n+1]*(1.0-weight) + daxis[n+2]*(weight)
            isPeak = (preSlope > threshold) and (postSlope < -1*threshold)
            zeroCrossings.append(isPeak)
        zeroCrossings.append(0)
        zeroCrossings = np.array(zeroCrossings)
        peaks = zeroCrossings
        
        
        
        #peaks = maxis
        #peaks = np.nonzero(np.logical_and(maxis, zeroCrossings))[0]
        allStrokes = np.nonzero(peaks)[0] + strokeStart
        strokes = []
        for n in range(1,len(allStrokes)):
            if allStrokes[n] - allStrokes[n-1] > window:
               strokes.append(allStrokes[n])
                
        return np.array(strokes)
    
    def getShaveStats(self, plot=False, outputDebug=False, axisToUse="Y", smoothing = 2, peakSmoothing=3, slope = 0.65, window=2, startUp=True):
        strokePeriods, rinsePeriods = self.getShaveIntervals(range=smoothing, axisToUse=axisToUse, startUp=startUp)
        strokesByPeriod = []
        if axisToUse == "X":
            axis = self.x
        elif axisToUse == "Y":
            axis = self.y
        elif axisToUse == "Z":
            axis = self.z
        else:
            axis = self.y
        for interval in strokePeriods:
            strokesByPeriod.append(self.findStrokes(interval.start, interval.end, threshold_slope=slope, smooth=peakSmoothing, window=window, axisToUse=axisToUse))
        strokeCountByPeriod = []
        for period in strokesByPeriod:
            strokes = period
            strokeCountByPeriod.append(len(strokes))
        rinseDurations = []
        strokePeriodDurations = []
        for interval in strokePeriods:
            strokePeriodDurations.append(interval.duration)
        for interval in rinsePeriods:    
            rinseDurations.append(interval.duration)
        if plot:
            plt.subplot(311)
            plt.plot(axis, color="blue")
            for interval in strokePeriods:
                plt.axvspan(interval.start, interval.end, color="gray", alpha=0.5)
            for period in strokesByPeriod:
                plt.scatter(period, np.array(axis)[np.array(period)], color="green")
            plt.ylabel(axisToUse + "-Axis Acceleration")
            plt.xlim(0, len(axis))
            plt.ylim(axis.min(), axis.max()*1.1)
            plt.subplot(312)
            plt.xlim(0, len(axis))
            plt.ylabel("Stroke Count")
            for n in range(0, len(strokePeriods)):
                interval = strokePeriods[n]
                plt.bar([interval.start], [strokeCountByPeriod[n]], interval.duration*self.sampleRate, color="blue")
                plt.text(interval.start - (interval.start - interval.end)/2, strokeCountByPeriod[n]/2, str(strokeCountByPeriod[n]), fontsize=12, color="white", ha='center',va='center', weight='semibold')
            plt.subplot(313)
            plt.xlim(0, len(self.z))
            plt.ylabel("Stroke/Rinse Durations (s)")
            plt.xlabel("Blue = Stroke | Red = Rinse")
            for period in strokePeriods:
                plt.bar([period.start], [period.duration], period.duration*self.sampleRate, color="blue")
            for period in rinsePeriods:
                plt.bar([period.start], [period.duration], period.duration*self.sampleRate, color="red")
            plt.show()
        if outputDebug:
            return (np.array(strokesByPeriod), np.array(strokeCountByPeriod), np.array(strokePeriods), np.array(rinsePeriods))
        else:
            return (np.array(strokeCountByPeriod), np.array(strokePeriodDurations), np.array(rinseDurations))
    
class edaStats:
    def __init__(self, log):
        self.log = log
        self.EDA = log.EDA()
        self.sampleRate = log.sampleRate
        self.computeStats()

    def computeStats(self):
        self.scrs = self.findSCRs()

    def peaks(self):
        peaks = []
        for s in self.scrs:
            peaks.append(s.peak)
        return np.array(peaks)
    
    def magnitudes(self):
        magnitudes = []
        for s in self.scrs:
            magnitudes.append(s.magnitude)
        return np.array(magnitudes)

    def durations(self):
        durations = []
        for s in self.scrs:
            durations.append(s.duration)
        return np.array(durations)

    def riseTimes(self):
        riseTimes = []
        for s in self.scrs:
            riseTimes.append(s.riseTime)
        return np.array(riseTimes)
    
    def decayTimes(self):
        decayTimes = []
        for s in self.scrs:
            decayTimes.append(s.decayTime)
        return np.array(decayTimes)
    
    def ppm(self, increment=60):
        ppm = []
        peaks = self.peaks()
        spm = int(self.sampleRate) * 60
        step = increment*int(self.sampleRate)
        for n in range(0, len(self.EDA) - len(self.EDA)%spm, step):
            ppm.append(np.sum(np.logical_and(peaks >= n, peaks < n+spm)))
        return np.array(ppm)
    
    def findSCRs(self, threshold=0.025, threshold2 = 0.001, slopeVsAccel=0.5, maxminRange=None):
        sr = int(self.sampleRate)
        if maxminRange == None:
            maxminRange = sr
        self.smoothEDA = scipy.ndimage.filters.gaussian_filter1d(self.EDA, maxminRange, order=0)
        SCREvents = []
        EDA = self.smoothEDA
        EDA1 = scipy.ndimage.filters.gaussian_filter1d(self.EDA, sr, order=1)
        EDA2 = scipy.ndimage.filters.gaussian_filter1d(self.EDA, sr, order=2)    
        isLocalMax = scipy.ndimage.maximum_filter1d( EDA, maxminRange*2+1) == EDA
        isLocalMin = scipy.ndimage.minimum_filter1d( EDA, maxminRange*2+1) == EDA
        isStart = scipy.ndimage.minimum_filter1d( EDA2, maxminRange*2+1) == EDA
        maxIndex = np.nonzero( isLocalMax )[0]
        minIndex = np.nonzero( isLocalMin ) [0]
        startIndex = np.nonzero( isStart ) [0]
        for maxima in maxIndex:
            a = SCR()
            a.peak = int(maxima)
            try:
                priorMin = np.max( minIndex[ minIndex < maxima ] )
                priorAccel = EDA2[priorMin:(maxima-1)]
                peakAccel = np.nonzero( priorAccel == np.max( priorAccel ) )[0][0] + priorMin
                
                priorSlope = EDA1[priorMin:maxima]
                peakSlope = np.nonzero( priorSlope == np.max( priorSlope ) )[0][0] + priorMin
                a.start = int( slopeVsAccel * float( peakSlope) + (1 - slopeVsAccel)* float(peakAccel))
            except exceptions.Exception as e:
                afflogger.exception(e)
                a.start = maxima
            try:
                postMin = np.min( minIndex[ minIndex > maxima ] )
                        
                postAccel = EDA2[(maxima+1):postMin]
                peakAccel = np.nonzero( postAccel == np.max( postAccel ))[0][-1] + (maxima+1)
                
                postSlope = EDA1[maxima:postMin]
                peakSlope = np.nonzero( postSlope == np.min( postSlope ))[0][-1] + (maxima+1)
                
                a.end = int( slopeVsAccel * float( peakSlope) + (1 - slopeVsAccel)* float(peakAccel))
            except:
                a.end = maxima
            a.riseTime = (a.peak - a.start)/sr
            a.decayTime = (a.end - a.peak)/sr
            a.area = np.sum(EDA[a.start:a.end])
            a.duration = (a.end - a.start)/sr
            a.magnitude = self.EDA[a.peak] - self.EDA[a.start]
            SCREvents.append(a)    
        return SCREvents

class SCR:
    start = None
    end = None
    peak = None
    startTime = None
    endTime = None
    peakTime = None
    magnitude = None
    area = None
    riseTime = None
    decayTime = None
    def toDict(self):
        if self.startTime is not None and self.endTime is not None and self.peakTime is not None:
            return {'startIndex': self.start, 'endIndex': self.end, 'peakIndex': self.peak, \
                    'startTimestamp': self.startTime, 'endTimestamp': self.endTime, 'peakTimestamp': self.peakTime, \
                    'magnitude': self.magnitude, 'area': self.area} 
        else:
            return {'startIndex': self.start, 'endIndex': self.end, 'peakIndex': self.peak, 'magnitude': self.magnitude, 'area': self.area}

