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

import time
import exceptions
from copy import *
import datetime as datetime
from qLogFile import *
from _utils import *
from filter import *
from visualize import *
from stats import *
from events import *

import ais.logger
afflogger = ais.logger.get_affdex_logger()


def combineEDAFiles(logs,pad=True):
	logfiles = sorted(logs, key=lambda x: x.startTime, reverse=False)
	if isOverlapInLogs(logfiles):
		afflogger.info("Error: Log time spans overlap, and cannot be combined")
	else:
		sampleRates = np.array(map(lambda x: x.sampleRate, logs))
		sr_target = sampleRates.min()
		for n in range(0, len(logfiles)):
			log = logfiles[n]
			if log.sampleRate > sr_target:
				logfiles[n] = log.downsample(sr_target)
		combinedData = deepcopy(logfiles[0])
		for n in range(1, len(logfiles)):
			log_n = logfiles[n]
			log_nminus1 = logfiles[n-1]
			if pad:
				padding = []
				dummySample = qDataPacket("0.0,0.0,0.0,0.0,0.0,0.0")
				numberOfSamplesNeeded = (log_n.startTime - log_nminus1.endTime).seconds*int(sr_target)
				for n in range(numberOfSamplesNeeded):
					padding.append(dummySample)
				combinedData.data.extend(padding)
				combinedData.data.extend(log_n.data)
		combinedData.endTime = combinedData.startTime + datetime.timedelta(seconds=(len(combinedData.data)/combinedData.sampleRate))
	return combinedData



def ppmForEvents(log,events, increment=10):
	ppm = dict()
	for event in events.events:
		start = event.startTime
		end = event.endTime
		startOffset = log.offsetForTime(start)
		endOffset = log.offsetForTime(end)
		ppm[event.comment] = computePPMForSubset(log, start=startOffset, end=endOffset, increment=increment)
	return ppm



