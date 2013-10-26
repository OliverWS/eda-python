# This Python file uses the following encoding: utf-8
#Copyright 2011 Affectiva, Inc.
#@author Oliver Wilder-Smith
#@collaborator Oliver Ernst Nowak '0xdeadbeef'

import os
import sys

from _utils import *
import datetime as datetime
from datetime import timedelta
import pytz as tz
import struct
import exceptions
import time
from events import Event

import numpy as np
import logging as log

class QParseError(exceptions.IOError):
    """Exception encountered from while parsing an EDA file"""
    pass

class qLogFile:
    """This class represents a collection of methods to work on qsensor
    log data.

    Methods include:
        + loadLogFile(self, filename, nice)
        + unpackSigned(self, data)
        + unpackUnsigned(self, data)
        + EDA(self)
        + vBat(self)
        + ACC_X(self)
        + ACC_Y(self)
        + ACC_Z(self)
        + Temperature(self)
        + edaSlice(self, start, end)
        + offsetForTime(self, time)
        + setEDA(self, new_data)
        + downsample(self, rate)
        + formatDate(self, x, pos=None)
        + cone_of_silence(self)
        + formatTime12HR(self, x, pos=None)
        + timeForOffset(self, offset)
        + formatTime(self, x, pos=None)
        + trim(self, start, end)
        + save(self, filename)

    """
    firmwareVersion = 0.0
    TIME_UNSET_TIME = datetime.datetime(1970,1,1,0,0,0).replace(tzinfo=tz.FixedOffset(0))
    
    def generateHeader(self, v2=False):
        if v2:
            header = ["Log File Created by Q Analytics - (c) 2011 Affectiva Inc.",
                               "File Version: 1.45",
                               "Firmware Version: %0.2f",
                               "UUID: %s",
                               "Sampling Rate: %d",
                               "Start Time: %s Offset:%s",
                               " %s ",
                               "---------------------------------------------------------\r\n"]
        else:
            header = ["Log File Created by Q Analytics - (c) 2011 Affectiva Inc.",
                               "File Version: 1.01",
                               "Firmware Version: %0.2f",
                               "UUID: %s",
                               "Sampling Rate: %d",
                               "Start Time: %s Offset:%s",
                               " Z-axis | Y-axis | X-axis | Battery | °Celsius | EDA(uS) ",
                               "---------------------------------------------------------\r\n"]
        header[2] = header[2]%(self.firmwareVersion)
        header[3] = header[3]%(self.sensorID)
        header[4] = header[4]%(int(self.sampleRate))
        if self.timeIsSet:
            tzoffset = int(self.startTime.utcoffset().total_seconds()/3600)
            tzstr = ("-" if tzoffset < 0 else "+") + "%02d"%(abs(tzoffset))
            header[5] = header[5]%(self.startTime.strftime("%Y-%m-%d %H:%M:%S"), tzstr)
        else:
            header[5] = "Start Time: Unset"

        if v2:
            header[6] = header[6]%(" | ".join(self.columnNames))
        return "\r\n".join(header)
    
    def __init__(self, file=None, filename="", verbose=True, nice=False):

        # try to open a valid file
        if isinstance(file, str) or isinstance(file, unicode):
            try:
                self.filename = os.path.split(file)[-1]
                file = open(file, "rb")
            except Exception as e:
                log.error("Error reading EDA File: %s"%(str(file)))
                log.exception(e)
        else:
            self.filename = filename
        self.data       = None
        self.verbose    = verbose
        self.startTime  = self.TIME_UNSET_TIME
        self.timeZone   = 0
        self.endTime    = 0
        self.sampleRate = 0
        self.sensorID   = ""
        self.headerText = ""
        self.markers    = []
        self.events     = []
        self.isBinary   = False
        self.num_samples = 0
        self.fileVersion = -1
        self.firmwareVersion = -1
        self.columnNames = ['Z-axis', 'Y-axis', 'X-axis', 'Battery', '\xc2\xb0Celsius', 'EDA(uS)']
        self.ncols = 6

        # ARRAY ACCESSOR CONSTANTS 
        self.FIELD_ACC_Z = 0
        self.FIELD_ACC_Y = 1
        self.FIELD_ACC_X = 2
        self.FIELD_BAT_V = 3
        self.FIELD_TEMP  = 4
        self.FIELD_EDA   = 5

        # load the data into the qLogFile instance
        if file != None:
            self.loadLogFile(file, nice)

    def loadLogFile(self, file_reference, nice):
        '''Load an EDA file and read its contents into the qLogFile data structure

        Params
        ------
        file_reference - a file object or a string URL to a file
        nice           - pretty formatting (True or False)'''

        if isinstance(file_reference, str):
            try:
                self.filename = os.path.split(file_reference)[-1]
                file_reference = open(file_reference, "rb")
            except Exception as e:
                log.error("Could not open file!")
                log.exception(e)
                return

        self.reader = file_reference
        rawData = self.reader.read()
        self.reader.close()        
        self.headerText = rawData.split("---------------------------------------------------------\r\n")[0]
        fileData = rawData.split("---------------------------------------------------------\r\n")[1]
        rawData = None

        # Parse header data from file
        self.read_header_data(self.headerText)

        # Validate the file or die
        self.validate_file()
        
        if self.isBinary:
            self.read_binary_data(fileData, nice)
        else: 
            self.read_text_data(fileData, nice)
        
        # Update the Start & End times for the log file
        self.update_logfile_times()

        # close the file reference
        file_reference.close()
    
    def read_header_data(self, file_data):
        headerLines = file_data.split("\n")

        for line in headerLines:
            try:
                if line.startswith("Sampling Rate:"):
                    self.sampleRate = float(line.split(':')[1].lstrip(" ").rstrip("\r\n"))
                elif line.startswith("UUID: "):
                    self.sensorID = line.split(':')[1].lstrip(" ").rstrip("\r\n")
                elif line.startswith("File Version: "):
                    self.fileVersion = float(line.split(':')[1].lstrip(" ").rstrip("\r\n"))
                    self.isBinary = self.fileVersion  >= 1.50
                elif line.startswith("Firmware Version: "):
                    self.firmwareVersion = float(line.split(':')[1].lstrip(" ").rstrip("\r\n"))
                    self.isBinary = self.fileVersion  >= 1.50
                elif line.startswith("Start Time:"):
                    #Time Format: 2011-04-14 07:13:40 Offset:-04
                    try:
                        self.startTimeStr = line.split("Offset")[0]
                        self.timeZone = int(line.split("Offset:")[1].rstrip("\r\n"))
                        timestr = line.split(': ')[1].lstrip(" ").split(" Offset")[0]
                        self.startTime = datetime.datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz.FixedOffset(self.timeZone*60))
                    except Exception as e:
                        log.error("Problem processing EDA File time string: %s" %(line))
                        log.error(e)
                        self.startTime = self.TIME_UNSET_TIME
                elif " | " in line:
                    colNames = line.lstrip(" ").rstrip(" ").rstrip("\r\n")
                    colNames = colNames.split(" | ")
                    self.columnNames = colNames
                    self.ncols = len(colNames)

            except Exception as e:
                log.error("Error processing EDA File: %s\nHeader is invalid\n%s"%(self.filename,e))
                log.exception(e)



    def validate_file(self):
        #Assert that all necessary attributes were parsed out of header and have reasonable values
        
        assert self.startTime >= self.TIME_UNSET_TIME, "Start Time invalid: %s"%(str(self.startTime))
        assert self.sampleRate >= 2, "Sample rate invalid: %s"%(str(self.sampleRate))
        assert len(self.sensorID) > 0, "Sensor ID invalid: Sensor ID must have length greater than 0. File type not supported."
        assert self.fileVersion > 1.0, "File version invalid or missing. File type not supported."
        assert self.firmwareVersion > 0, "Firmware version invalid or missing. File type not supported."
        #assert self.sampleRate <=32, "Sample rate invalid: %s"%(str(self.sampleRate))
        
    def read_text_data(self, file_data, nice=False):
        index = 0
        lines = 0
        fields= 6
        progress_complete = 0.0

        if self.verbose:
            print "Reading text file"

        # split the data along new-line delimiters
        data_packets = file_data.split('\r\n')
        lines = len(data_packets)

        # allocate the numpy array given the sample size
        sample_array = np.zeros( (lines, fields), 'float32')
       
        # iterate through each line
        # split the samples into fields and add them to the numpy array
        for line in data_packets:
            try:
                line = line.rstrip('\r\n')
            
                if line == '':
                    log.info('>> Encountered a blank line on line #%i in (headless) fileData' % (index))
                elif line == ',,,,,':
                    self.markers.append(index)
                else:
                    samples = line.split(',')
                    if len(samples) != self.ncols: 
                        continue
                    else:
                        for i,field in enumerate(samples):
                            # handle fields which may contain a string
                            if field == 'N/A':
                                sample_array[index, i] = -1
                            else:
                                sample_array[index, i] = field
                    
                        if nice and (index % (10 * self.sampleRate)) == 0:
                            time.sleep(0.001)
                    
                        index += 1
                    
                # update the progress bar and recurse the percent completed for
                # the next iteration
                if self.verbose:
                    progress_complete = self.update_progress_bar(index, lines, progress_complete)

            except Exception as e:
                log.error("> Problem reading packet on line #%i in (headless) fileData" % (index + 1) )
                log.exception(e)
                self.lock_data()
                self.num_samples = len(sample_array)
                break
        
        # copy sample_array over to class data structure 
        self.data = sample_array

        # make numpy array immutable since all downstream calculations
        # assume the data is invariable
        self.lock_data()  

        self.num_samples = len(self.data)
    
    def read_binary_data(self, file_data, nice=False):
        index = 0
        lines = 0
        fields = 6
        progress_complete = 0.0
        EOL = b'\xe5\xe2'

        if self.verbose:
            print "Decoding binary file Version "+ str(self.fileVersion)
        
        # split the data along new-line delimiters
        data_packets = file_data.split(EOL)
        lines = len(data_packets)

        # allocate the numpy array given the sample size
        self.data = np.zeros( (lines, fields), 'float32')

        # initialize field vars
        acc_x = 0
        acc_y = 0
        acc_z = 0
        bat_v = 0
        temp  = 0
        eda   = 0

        # iterate through array and unpack the binary data
        for line in data_packets:
            try:
                # check for blank lines that could occur at EOF and log them
                if len(line) == 0:
                    log.info("> Encountered a blank line at #%i of (headless) binData - this is most likely EOF" % (index))
                    self.lock_data()
                    self.num_samples = len(self.data)
                    break
                
                samples = struct.unpack(">HHHHHH", line)
                
                # using unrolled loop for speed and code readability
                acc_z = self.unpackSigned(samples[self.FIELD_ACC_Z])
                acc_y = self.unpackSigned(samples[self.FIELD_ACC_Y])
                acc_x = self.unpackSigned(samples[self.FIELD_ACC_X])
                bat_v = self.unpackUnsigned(samples[self.FIELD_BAT_V]) * 10.0
                temp  = self.unpackSigned(samples[self.FIELD_TEMP])
                eda   = self.unpackUnsigned(samples[self.FIELD_EDA])

                if eda >= 999 and acc_x <= -999.0:
                    self.markers.append(index)
                else:
                    self.data[index, self.FIELD_ACC_Z] = acc_z
                    self.data[index, self.FIELD_ACC_Y] = acc_y
                    self.data[index, self.FIELD_ACC_X] = acc_x
                    self.data[index, self.FIELD_BAT_V] = bat_v
                    self.data[index, self.FIELD_TEMP] = temp
                    self.data[index, self.FIELD_EDA] = eda

                    index += 1
                    
                    if nice and (index % (10 * int(self.sampleRate))) == 0:
                        time.sleep(0.001)
                
                # update the progress bar and recurse the percent completed for
                # the next iteration
                if self.verbose:
                    progress_complete = self.update_progress_bar(index, lines, progress_complete)

            except Exception as e:
                log.error("> Problem reading packet in (headless) binData")
                log.exception(e)
                self.lock_data()
                self.num_samples = len(self.data)
                break

        # make numpy array immutable since all downstream calculations
        # assume the data is invariable
        self.lock_data()

        self.num_samples = len(self.data)
    
    def update_logfile_times(self):
        try:
            self.endTime = self.startTime + datetime.timedelta(seconds=(self.num_samples/self.sampleRate))
        except exceptions.Exception as e:
            self.endTime = self.startTime
            log.exception(e)
            log.error(e)
        if self.verbose:
            log.info("Start: " + self.startTime.isoformat(' '))
            log.info("End: " + self.endTime.isoformat(' '))

    def update_progress_bar(self, progress, progress_total, percent_complete):
        if (float(progress) / float(progress_total)) - percent_complete > 0.01:
            percent_complete = float(progress) / float(progress_total)
            
            if percent_complete >= .99: 
                percent_complete = 1.
            
            progressBar(percent_complete)
        
        return percent_complete

    
    def lock_data(self):
        self.data.flags.writeable = False

    def unlock_data(self):
        self.data.flags.writeable = True

    def unpackSigned(self, data):
        isNegative = testBit(data, 15) > 0
        dotOffset =  (data >> 13) & 3 #Rotate and mask w/ 0x0003
        value = (data & int("0x03FF", 16))/1000.0
        if isNegative:
            return truncate(round((-1.0) * value * (pow(10, dotOffset)), 3 - dotOffset), 3 - dotOffset)
        else:
            return truncate(round(value * (pow(10, dotOffset)), 3 - dotOffset), 3 - dotOffset)
    
    def unpackUnsigned(self, data):
        dotOffset =  (data >> 14) & 3 #Rotate and mask w/ 0x0003
        value = (data & int("0x3FFF", 16))/10000.0
        return truncate(round(value * pow(10, dotOffset), 4 - dotOffset), 4 - dotOffset)

    def EDA(self):
        return [ self.data[i, self.FIELD_EDA] for i in range(self.num_samples) ]
    
    def vBAT(self):
        return [ self.data[i, self.FIELD_BAT_V] for i in range(self.num_samples) ]
    
    def ACC_X(self):
        return [ self.data[i, self.FIELD_ACC_X] for i in range(self.num_samples) ]

    def ACC_Y(self):
        return [ self.data[i, self.FIELD_ACC_Y] for i in range(self.num_samples) ]
    
    def ACC_Z(self):
        return [ self.data[i, self.FIELD_ACC_Z] for i in range(self.num_samples) ]

    def Temperature(self):
        return [ self.data[i, self.FIELD_TEMP] for i in range(self.num_samples) ]
    
    def addColumn(self, newData, name=""):
        if self.data == None:
            self.data = np.zeros( (len(newData), 6), 'float32')
            self.num_samples = len(self.data)
        self.unlock_data()
        if name == "":
            name = "Field%02d"%(self.ncols + 1)
        newData = np.array(newData)
        if len(newData) != self.num_samples:
            raise IndexError("New Data length (%d) not equal to current data length (%d)"%(len(newData), self.num_samples ))

        self.data = np.column_stack([self.data,newData])
        self.columnNames.append(name)
        self.ncols = self.data.shape[1]
        self.lock_data()
        self.update_logfile_times()

    def setData(self,field, newData):
        
        if self.data == None:
            self.data = np.zeros( (len(newData), 6), 'float32')
            self.num_samples = len(self.data)
        self.unlock_data()
        if field.lower() == "eda":
            field = self.FIELD_EDA
        elif field.lower() == "accz" or field.lower() == "z":
            field = self.FIELD_ACC_Z
        elif field.lower() == "accy" or field.lower() == "y":
            field = self.FIELD_ACC_Y
        elif field.lower() == "accx" or field.lower() == "x":
            field = self.FIELD_ACC_X
        elif field.lower() == "temp" or field.lower() == "temperature":
            field = self.FIELD_TEMP
        elif field.lower() == "vbat" or field.lower() == "battery":
            field = self.FIELD_BAT_V
        else:
            raise KeyError("Field %s is not a valid data field"%(field))
        if len(newData) == self.num_samples:
            for i in range(self.num_samples):
                self.data[i, field] = newData[i]
        # lock the backing numpy array 
        self.lock_data()
        self.update_logfile_times()
    
    def clone(self):
        from copy import deepcopy
        cloneLog = deepcopy(self)
        return cloneLog

    @property
    def timeIsSet(self):
        return (self.startTime != self.TIME_UNSET_TIME)
    
    def edaSlice(self,start,end):
        '''Return a slice of the EDA data (non-destructive)

        Params
        ------
        start - an index offset OR a datetime object
        end   - an index offset OR a datetime object

        Returns:
            a slice of the EDA data as a List'''

        isTime = (isinstance(start,datetime.datetime) and isinstance(end, datetime.datetime))
        
        # convert to sample offset if params are datetime objects
        if isTime:
            start = self.offsetForTime(start)
            end = self.offsetForTime(end)
        
        if end > start:
            return self.EDA()[start : end]
        else:
            raise IndexError("End offset is less than Start offset.")

    def offsetForTime(self, time):
        '''Return an offset in EDA data, given a datetime object

        Params
        ------
        time - a datetime object

        Returns:
            an index offset into the EDA data'''

        totalSeconds = 0
        if isinstance(time,datetime.datetime):
           if self.startTime > time:
               raise IndexError("Given time is earlier than log file start time. No dava available!")                  
           elif time > self.endTime:
               raise IndexError("Given time is later than log file end time. No data available!")
           else:   
               delta = time - self.startTime
               totalSeconds = (delta.microseconds + (delta.seconds + delta.days * 24 * 3600) * 10**6) / 10**6
        
               return int(math.floor(totalSeconds)*int(self.sampleRate))
        else:
           raise TypeError("the given parameter is not a datetime object!")

    def setEDA(self, newData):
        # make backing numpy array writeable 
        self.unlock_data()

        if len(newData) == self.num_samples:
            for i in range(self.num_samples):
                self.data[i, self.FIELD_EDA] = newData[i]
        
        # lock the backing numpy array 
        self.lock_data()
    
    def downsample(self, rate):
        '''Compress the EDA data to a smaller sampling rate.

        Params
        ------
        rate - the desired sampling rate

        Returns:
            a qLogFile object with its data compressed to the specified rate'''

        import scipy
        import scipy.signal

        if rate >= self.sampleRate or rate < 0:  
           raise ValueError("Rate param is greater than the sample rate OR the rate is less than 0")
        else:
            newLength = self.num_samples * (float(rate)/self.sampleRate)
            newLength = int(round(newLength))
            eda = scipy.signal.resample(self.EDA(), newLength)
            accx = scipy.signal.resample(self.ACC_X(), newLength)
            accy = scipy.signal.resample(self.ACC_Y(), newLength)
            accz = scipy.signal.resample(self.ACC_Z(), newLength)
            temp = scipy.signal.resample(self.Temperature(), newLength)
           
            # number of fields for the numpy array
            fields = 6
            
            # make the backing numpy array writeable
            self.unlock_data()

            # clear the numpy array and re-init with new size
            self.data = np.zeros( (newLength, fields), 'float16')
            for i in range(0, len(self.markers)):
                self.markers[i] = int(round(self.markers[i]*(float(rate)/self.sampleRate)))
            # parse the downsampled packets and add them to the numpy array
            for i in range(0, newLength):
                self.data[i, self.FIELD_ACC_Z] = accz[i]
                self.data[i, self.FIELD_ACC_Y] = accy[i]
                self.data[i, self.FIELD_ACC_X] = accx[i]
                self.data[i, self.FIELD_BAT_V] = -1.
                self.data[i, self.FIELD_TEMP]  = temp[i]
                self.data[i, self.FIELD_EDA]   = eda[i]

            # lock the numpy array to make the data immutable    
            self.lock_data()

            # update the sample length
            self.num_samples = len(self.data)
            self.sampleRate = float(rate)

            return self 
    
    def formatDate(self, x):
        delta = 1/self.sampleRate
        return (self.startTime + datetime.timedelta(seconds=x*delta)).strftime("%H:%M:%S \n %m/%d/%Y")
    
    def cone_of_silence(self):
        print 'Protected by a cone of silence.'

    def formatTime12HR(self,x):
        delta = 1.0/self.sampleRate
        t = x*delta
        return (self.startTime + datetime.timedelta(seconds=t)).strftime("%I:%M:%S %p")
    
    def timeForOffset(self,offset):
        return self.startTime + datetime.timedelta(seconds=(offset/self.sampleRate))
    
    def formatTime(self, x):
        delta = 1/self.sampleRate
        return (self.startTime + datetime.timedelta(seconds=x*delta)).strftime("%H:%M:%S")

    def trim(self,start,end):
        '''Trim an EDA file given a start and end <OFFSET> or <DATETIME object>.

        Params
        ------
        start - starting index offset OR start time as datetime object
        end   - ending index offset OR end time as datetime object

        Returns:
            self trimmed
        '''

        isTime = (isinstance(start,datetime.datetime) and isinstance(end, datetime.datetime))
        
        # convert to sample offset if params are datetime objects
        if isTime:
            start = self.offsetForTime(start)
            end = self.offsetForTime(end)
        
        self.unlock_data()
        self.data = self.data[start:end, :]
        self.lock_data()

        self.headerText = self.headerText.replace("Start Time: "+self.startTime.strftime("%Y-%m-%d %H:%M:%S"), "Start Time: "+self.timeForOffset(start).strftime("%Y-%m-%d %H:%M:%S"))
        self.startTime = self.timeForOffset(start)
        self.endTime = self.startTime + datetime.timedelta(seconds=(self.num_samples/self.sampleRate))
        self.markers = map(lambda x: x - start, [x for x in self.markers if x > start and x < end])
        if hasattr(self,"peakList"):
            self.peakList = [x for x in self.peakList if x["start"] > start and x["end"] < end]
        if hasattr(self,"lability"):
            self.lability = [x for x in self.lability if x["start"] > start and x["end"] < end]
        self.num_samples = len(self.data)

        return self
    
    def updateStartTime(self, newtime):
        '''Update the EDA Start Time field with a new start time.

        Params
        ------
        newtime - a datetime object containing the new start time'''
        
        if isinstance(newtime,datetime.datetime):
            self.headerText = self.headerText.replace("Start Time: "+self.startTime.strftime("%Y-%m-%d %H:%M:%S"), "Start Time: "+newtime.strftime("%Y-%m-%d %H:%M:%S"))
            self.startTime = newtime
            self.endTime = self.startTime + datetime.timedelta(seconds=(self.num_samples/self.sampleRate))
        else:
            raise TypeError("the parameter given is not a datetime object!")

    def save(self, filename):
        '''Saves data to disk as an EDA file.

        Params
        ------
        filename - the filename of the saved data'''
           
        self.writer = open(filename,"wb")
        self.writer.write(self.generateHeader())
        
        # initialize progress complete counter
        progress_complete = 0.0
        index = 0

        # initialize loop variables outside loop for speed
        acc_z = 0.0
        acc_y = 0.0
        acc_x = 0.0
        bat_v = 0.0
        temp  = 0.0
        eda   = 0.0

        for n in range(self.num_samples):
            try:
                acc_z = self.data[n, self.FIELD_ACC_Z]
                acc_y = self.data[n, self.FIELD_ACC_Y]
                acc_x = self.data[n, self.FIELD_ACC_X]
                bat_v = self.data[n, self.FIELD_BAT_V]
                temp  = self.data[n, self.FIELD_TEMP]
                eda   = self.data[n, self.FIELD_EDA]

                line = "%f,%f,%f,%f,%f,%f\r\n" %(acc_z, acc_y, acc_x, bat_v, temp, eda)
                self.writer.write(line)

                index += 1
                
                if n in self.markers:
                    self.writer.write(",,,,,\r\n")
            
                if self.verbose:
                    progress_complete = self.update_progress_bar(index, self.num_samples, progress_complete)
            
            except exceptions.Exception as e:
                log.error("Problem outputting data packet:\n%s"%(str(e)))
                log.exception(e)

        self.writer.close()

    def save2(self, filename):
        '''Saves data to disk as an EDA file using new channel format

        Params
        ------
        filename - the filename of the saved data'''
           
        self.writer = open(filename,"wb")
        self.writer.write(self.generateHeader(v2=True))
        
        # initialize progress complete counter
        progress_complete = 0.0
        index = 0

        # initialize loop variables outside loop for speed

        for n in range(self.num_samples):
            try:

                line = "%s\r\n"%(",".join(map(self.autoconvert, self.data[n, :])))
                self.writer.write(line)

                index += 1
                
                # if n in self.markers:
                #     self.writer.write(",,,,,\r\n")
            
                if self.verbose:
                    progress_complete = self.update_progress_bar(index, self.num_samples, progress_complete)
            
            except exceptions.Exception as e:
                log.error("Problem outputting data packet:\n%s"%(str(e)))
                log.exception(e)

        self.writer.close()

    def autoconvert(self,s):
        try:
            if type(s) == float:
                return "%f"%(s)
            elif type(s) == str:
                return "\"%s\""%(s)
            elif type(s) == datetime.datetime:
                return s.isoformat()
            else:
                return str(s)
        except Exception as e:
            log.error("Problem converting value: %s\n%s", str(s), str(e))
            return str(s)

    def toJSON(self):
        '''Creates a JSON representation of the eda file'''
        import json
        j = dict()
        j["startTime"] = self.startTime.isoformat()
        j["endTime"] = self.endTime.isoformat()
        j["eda"] = map(float, self.EDA())
        j["range"] = [float(np.min(self.EDA())),float(np.max(self.EDA()))]
        j["events"] = []
        if len(self.events) > 0:
            for event in self.events:
                e = dict()
                e["start"] = self.offsetForTime(event.startTime)
                e["startTime"] = event.startTime.isoformat()

                if event.isRange():
                    e["end"] = self.offsetForTime(event.endTime)
                    e["endTime"] = event.endTime.isoformat()
                else:
                    e["end"] = -1
                e["label"] = event.note
                j["events"].append(e)
        for m in self.markers:
            e = dict()
            e["start"] = m
            e["startTime"] = self.timeForOffset(m).isoformat()
    
            e["end"] = -1
            e["label"] = ""
            j["events"].append(e)

        j["hours"] = []
        for hour in np.arange(self.startTime.hour+1, self.endTime.hour+1):
            t = self.startTime.replace(hour=hour,minute=0,second=0)
            loc = self.offsetForTime(t)
            j["hours"].append({"label":t.strftime("%I:%M %p"),"offset":loc})
        j["fps"] = self.sampleRate
        return json.dumps(j, indent=4)
        
    
    def toString(self):
        '''Creates a String-based List of the EDA data, with accompanying 
        Header info.

        Returns:
            An EDA file.'''

        output = self.generateHeader()
        progress_complete = 0.0
        index = 0

        # initialize loop variables outside loop for speed
        acc_z = 0.0
        acc_y = 0.0
        acc_x = 0.0
        bat_v = 0.0
        temp  = 0.0
        eda   = 0.0

        for n in range(self.num_samples):
            try:
                acc_z = self.data[n, self.FIELD_ACC_Z]
                acc_y = self.data[n, self.FIELD_ACC_Y]
                acc_x = self.data[n, self.FIELD_ACC_X]
                bat_v = self.data[n, self.FIELD_BAT_V]
                temp  = self.data[n, self.FIELD_TEMP]
                eda   = self.data[n, self.FIELD_EDA]
                
                line = "%f,%f,%f,%f,%f,%f\r\n" %(acc_z, acc_y, acc_x, bat_v, temp, eda)

                output += line
                index  += 1

                if n in self.markers:
                    output += ",,,,,\r\n"

                if self.verbose:
                    progress_complete = self.update_progress_bar(index, self.num_samples, progress_complete)

            except exceptions.Exception as e:
                log.error("Problem outputting data packet:\n%s"%(str(e)))
                log.exception(e)

        return output

    @property
    def eda(self):
        return np.array(self.EDA())
    @property
    def acc_x(self):
        return np.array(self.ACC_X())
    @property
    def acc_y(self):
        return np.array(self.ACC_Y())
    @property
    def acc_z(self):
        return np.array(self.ACC_Z())
    @property
    def temperature(self):
        return np.array(self.Temperature())
    @property
    def vbatt(self):
        return np.array(self.vBAT())

    @property
    def timestamps(self):
        return np.arange(self.startTime,self.endTime,timedelta(seconds=(1./self.sampleRate))).astype(object)
    def secondsForOffset(self, offset):
        '''Returns an offset in seconds of the EDA data, given an index value

        Params
        ------
        offset - the index offset used to calculate the number of seconds 

        Returns:
            the number in seconds calculated from the given offset'''

        return (self.timeForOffset(offset) - self.startTime).seconds
    
    def offsetForSeconds(self, seconds):
        '''Returns an index offset of the EDA data, given a time in seconds.

        Params
        ------
        seconds - the number of seconds used to calculate the offset

        Returns:
            an index offset of the EDA data'''

        return self.offsetForTime(self.startTime + timedelta(seconds=seconds))

    def exportToCSV(self):
        '''Export to CSV.

        Returns:
            A List containing the EDA data in CSV format.'''

        output = ",".join(["Time","Z","Y","X","Temperature","EDA", "Markers"]) + "\r\n"
        progress_complete = 0.0
        index = 0

        # initialize loop variables outside loop for speed
        acc_z = 0.0
        acc_y = 0.0
        acc_x = 0.0
        bat_v = 0.0
        temp  = 0.0
        eda   = 0.0
        print 'n_samples : ', self.num_samples

        for n in range(self.num_samples):
            try:
                acc_z = self.data[n, self.FIELD_ACC_Z]
                acc_y = self.data[n, self.FIELD_ACC_Y]
                acc_x = self.data[n, self.FIELD_ACC_X]
                bat_v = self.data[n, self.FIELD_BAT_V]
                temp  = self.data[n, self.FIELD_TEMP]
                eda   = self.data[n, self.FIELD_EDA]

                line = "%s,%f,%f,%f,%f,%f,%f,%i\r\n" %(self.timeForOffset(n).strftime("%H:%M:%S.%f"),acc_z, acc_y, acc_x, bat_v, temp, eda, (1 if n in self.markers else 0))
                
                output += line
                index  += 1

                if self.verbose:
                    progress_complete = self.update_progress_bar(index, self.num_samples, progress_complete)

            except exceptions.Exception as e:
                log.error("Problem outputting data packet:\n%s"%(str(e)))
                log.exception(e)

        return output

