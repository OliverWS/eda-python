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
    
import ais.logger
afflogger = ais.logger.get_affdex_logger()

def correlate(log, log2):
    import matplotlib.units as units
    import matplotlib.dates as dates
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    if log.startTime > log2.endTime or log2.startTime > log.endTime:
        afflogger.info("No overlap; correlation not possible")
    else:
        signal1 = log.EDA()
        signal2 = log2.EDA()
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
        correlation, confidence = scipy.stats.pearsonr(signal1, signal2)
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
        track = np.arange(0, len(signal1)-1-spm, spm)
        #plt.plot(corr)
        plt.bar(track, corr, spm)
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
        return correlation


def removeBaseline(log):
    signal = log.EDA()
    nq = log.sampleRate/2.0
    fcutoff = 0.001/nq
    n=1024
    filterType = False
    filter = scipy.signal.firwin(n, cutoff=fcutoff, window = 'hamming')
    if not filterType:
        filter = -filter
        filter[n/2] = filter[n/2] + 1
    return scipy.signal.lfilter(filter, 1, signal)

def lowpass(log, cutoff):
    signal = log.EDA()
    nq = log.sampleRate/2.0
    fcutoff = cutoff/nq
    n=1024
    filterType = False
    filter = scipy.signal.firwin(n, cutoff=fcutoff, window = 'hamming')
    log.setEDA(scipy.signal.lfilter(filter, 1, signal))
    return log

def getTonic(signal, sr):
    return scipy.signal.medfilt(signal, sr*60 + 1)

    
def findSleep(log):
    x = np.array(log.ACC_X())
    y = np.array(log.ACC_Y())
    z = np.array(log.ACC_Z())
    
    work = sqrt(x*x + y*y + z*z)
    spm = int(60*log.sampleRate)
    sr = int(log.sampleRate)
    work_std = []
    
    for n in range(0, len(work) - spm, spm):
        work_std.append(np.std(work[n:n+spm]))
    
    work_std_smooth = scipy.ndimage.filters.median_filter(work_std, 5*sr)
    isSleep = work_std_smooth < 0.05
    sleepStart = []
    sleepEnd = []
    for n in range(1, len(isSleep)):
        if not isSleep[n-1] and isSleep[n]:
            sleepStart.append(n*spm)
        elif isSleep[n-1] and not isSleep[n]:
            sleepEnd.append(n*spm)
    return (sleepStart, sleepEnd)
    


def lowpassSignal(data,sr, cutoff):
    signal = deepcopy(data)
    nq = sr/2.0
    fcutoff = cutoff/nq
    n=1024
    filterType = False
    filter = scipy.signal.firwin(n, cutoff=fcutoff, window = 'hamming')
def filter(log):
    afflogger.info("Filtering")
    signal = log.EDA()
    nq = log.sampleRate/2.0
    fcutoff = float(sys.argv[3])/nq
    #afflogger.info("nq = " + str(nq) + " fcutoff = " + str(fcutoff))
    n=1024
    filterType = True
    if len(sys.argv) > 4:
        if sys.argv[4] == "hp":
            filterType = False
    filter = scipy.signal.firwin(n, cutoff=fcutoff, window = 'hamming')
    if not filterType:
        filter = -filter
        filter[n/2] = filter[n/2] + 1
    log.setEDA(scipy.signal.lfilter(filter, 1, signal))
    log.save("Filtered"+sys.argv[1])
    afflogger.info("Done")

def medfilter(log):
    afflogger.info("Median filtering")
    log.setEDA(scipy.signal.medfilt(log.EDA(), log.sampleRate*3 + 1))
#     if sys.argv[2] == "--medfilter":
#         filename = "MedianFiltered_" + log.filename
#         if os.path.isdir(sys.argv[1]):
#             filename = os.path.join(sys.argv[1], filename)
#         log.save(filename)
    afflogger.info("Done")
    return log
