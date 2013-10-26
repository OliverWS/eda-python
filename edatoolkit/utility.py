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
from copy import deepcopy

import ais.logger
afflogger = ais.logger.get_affdex_logger()



def cropEDAFile(logfile, start, end, filename=None, path=""):
	'''
		utility.cropEDAFile:
			input:
				logfile	= qLogFile to be cropped (note that cropping returns a copy of this object)
				start = String with time to start crop at in form "%m-%d-%Y %H:%M:%S" (i.e "07-01-2011 14:28:32")
				end = String with time to stop crop at in form "%m-%d-%Y %H:%M:%S" (i.e "07-01-2011 14:45:01")
			optional:
				filename= <desired output filename>
				path= <desired output path>
	'''
	log = deepcopy(logfile)
	startTime = datetime.datetime.strptime(start, "%m-%d-%Y %H:%M:%S").replace(tzinfo=log.startTime.tzinfo) if type(start) == str else start
	endTime = datetime.datetime.strptime(end, "%m-%d-%Y %H:%M:%S").replace(tzinfo=log.startTime.tzinfo) if type(end) == str else end
	if (startTime >= log.startTime and startTime < log.endTime) and (endTime > log.startTime and endTime <= log.endTime):
		afflogger.info("Cropping " + log.filename + " to " + startTime.strftime("%m-%d-%Y %H:%M:%S") + "\t:\t" + endTime.strftime("%m-%d-%Y %H:%M:%S"))
		startIndex = int(log.offsetForTime(startTime))
		endIndex = int(log.offsetForTime(endTime))
		log = log.trim(startIndex, endIndex)
		if filename == None:
			filename = log.filename.replace(".eda", "_Cropped.eda")
		log.save(os.path.join(path, filename))
		afflogger.info("Cropped file saved to " + os.path.join(path, filename))
	else:
		afflogger.info("ERROR: There was a problem cropping %s to %s"%(log.filename, filename))
		afflogger.info("File has START: %s and END: %s"%(startTime.strftime("%m-%d-%Y %H:%M:%S"), endTime.strftime("%m-%d-%Y %H:%M:%S") ))
		afflogger.info("Can't crop to START: %s and END: %s"%(log.startTime.strftime("%m-%d-%Y %H:%M:%S"), log.endTime.strftime("%m-%d-%Y %H:%M:%S") ))

#
#def help():
#	print "Supported commands:"
#	print "--crop:"
#	print "\tCrops and EDA file to specified start and end times"
#	print "\tUsage: EDA_Utilities.py <filename> --crop <start> <end> <desired output filename (optional)>"
#	print "\tExample: EDA_Utilities.py example.eda --crop '04-27-2011 15:53:00' '04-27-2011 16:35:45' example_cropped.eda"
#	print "--convert:"
#	print "\tConverts a binary EDA file to text-based (CSV) formatted file"
#	print "\tUsage: EDA_Utilities.py <filename> --convert <desired output filename (optional)>"
#	print "\tExample: EDA_Utilities.py example.eda --convert example_converted.eda"
#
#if __name__ == "__main__":
#	print "EDA Utility Version " + __version__()
#	print "Use option --help for a list of available commands"
#	
#	if sys.argv[1] == "--help":
#		help()
#	elif len(sys.argv) >= 3:
#		log = qLogFile(sys.argv[1], verbose=False)
#		function = sys.argv[2]
#		if function == "--crop":
#			if len(sys.argv) > 4:
#				start = sys.argv[3]
#				end = sys.argv[4]
#				path = os.path.split(sys.argv[1])[0]
#				if len(sys.argv) > 5:
#					outputFilename = sys.argv[5]
#					cropEDAFile(log, start, end, filename=outputFilename, path=path)
#				else:
#					cropEDAFile(log, start, end, path=path)
#			else:
#				print "Error: not enough information supplied for --crop"
#				help()
#		elif function == "--convert":
#			log = qLogFile(sys.argv[1], verbose=False)
#			path = os.path.split(sys.argv[1])[0]
#			print "Converting " + log.filename + " to text/csv format"
#			if len(sys.argv) == 4:
#				outputFilename = sys.argv[5]
#				log.save(os.path.join(path, outputFilename))
#			else:
#				log.save(os.path.join(path, log.filename.replace(".eda", "_Converted.eda")))
#				print "Converted file saved to: " + os.path.join(path, log.filename.replace(".eda", "_Converted.eda"))
#	else:
#		help()

	