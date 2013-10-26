from xml.dom import minidom
from xml.sax.saxutils import escape
import datetime
from datetime import timedelta
import time
import exceptions

class Event:
    start = 0
    end = None
    note = ""
    filename = ""
    isDeviceEvent = False
    file=None
    
    def __init__(self,start,end=None,note="", isDeviceEvent=False, file=None):
        if file != None:
            self.file = file
        if type(start) == datetime.datetime:
            start = self.getTimestamp(start)
        if type(end) == datetime.datetime:
            end = self.getTimestamp(end)
        self.start = start
        self.end = end
        self.note = note
        self.isDeviceEvent = isDeviceEvent
    
    def startOffset(self):
        if self.file != None:
            return self.file.offsetForTime(self.file.startTime + timedelta(seconds=self.start))
        else:
            return 0
    def endOffset(self):
        if self.file != None:
            return self.file.offsetForTime(self.file.startTime + timedelta(seconds=self.end))
        else:
            return 0
    @property
    def startSeconds(self):
        try:
            return (time.mktime(self.startTime.timetuple()) - time.mktime(self.file.startTime.timetuple()))
        except exceptions.Exception as e:
            return self.start
    @property
    def endSeconds(self):
        try:
            return (time.mktime(self.endTime.timetuple()) - time.mktime(self.file.startTime.timetuple()))
        except exceptions.Exception as e:
            return self.end

    @property
    def endTime(self):
        if self.end != None:
            if self.file != None:
                return self.file.startTime + timedelta(seconds=self.end)
            else:
                return datetime.datetime.fromtimestamp(self.end)
        else:
            return None
    @property
    def startTime(self):
        if self.start != None:
            if self.file != None:
                return self.file.startTime + timedelta(seconds=self.start)
            else:
                return datetime.datetime.fromtimestamp(self.start)
        else:
            return None
    @property
    def duration(self):
        if self.isRange():
            return (self.endTime - self.startTime)
        else:
            return 0

    def isRange(self):
        if self.end != None:
            return True
        else:
            return False
    def getTimestamp(self,dt):
        if self.file != None:
            return time.mktime(dt.astimezone(self.file.startTime.tzinfo).timetuple()) - time.mktime(self.file.startTime.timetuple())
        else:    
            return time.mktime(dt.astimezone(self.file.startTime.tzinfo).timetuple())
    def toxml(self):
        start_ms = int(round(self.startSeconds*1000.0))
        device = "true" if self.isDeviceEvent else "false"
        if self.isRange():
            end_ms = int(round(self.endSeconds*1000.0))
            node = "<Event device='%s' offset='%d' partner='%d'><Notes>%s</Notes></Event>"%(device, start_ms, end_ms, escape(self.note))
            node2 = "<Event device='%s' offset='%d' partner='%d'><Notes>%s</Notes></Event>"%(device, end_ms, start_ms,escape(self.note)) #Add pairing node that's identical. Don't ask me why.
            return (minidom.parseString(node).firstChild,minidom.parseString(node2).firstChild)  #To avoid having it stick an XML DocType tag at the top

        else:
            node = "<Event device='%s' offset='%d'><Notes>%s</Notes></Event>"%(device, start_ms,escape(self.note))
            return minidom.parseString(node).firstChild
        



'''
def importEvents(eventsFile, date = None):
    ef = EventList(eventsFile, date)
    return ef



class Event:
    def __init__(self, startTime, comment=None, endTime=None, file = None):
        inputtype = ""
        if file != None:
            self.file = file
        if comment != None:
            self.comment = str(comment)
        if type(startTime) == "string":
            self.startTime = datetime.datetime.strptime(startTime, "%m-%d-%Y %H:%M:%S")
            inputtype="string"
        else:
            try:
                self.startTime = startTime
                inputtype="datetime"
            except:
                self.startTime=None
                print "Problem loading event: no valid startTime given"
        if endTime != None:
            if inputtype == "datetime":
                if endTime > self.startTime:
                    self.endTime = endTime
                else:
                    print "Problem loading event: end time earlier than start time"
                    self.endTime = None
            elif inputtype == "string":
                parsedEndTime = datetime.datetime.strptime(endTime, "%m-%d-%Y %H:%M:%S")
                if parsedEndTime > self.startTime:
                    self.endTime = parsedEndTime
                else:
                    print "Problem loading event: end time earlier than start time"
                    self.endTime = None
            else:
                self.endTime=None
                print "Given endtime not valid!"
        if self.endTime != None and self.startTime != None and self.startTime < self.endTime:
            self.isRange = True
        else:
            self.isRange = False

class EventList:
    def __init__(self, eventfile, date=None):
        f = open(eventfile)
        file = f.read()
        header = file.split("------------------------------\r\n")[0].split("\r\n")
        events = file.split("------------------------------\r\n")[1].split("\r\n")
        print "Events: " + str(events)
        self.events = []
        self.date = None
        self.dateString = ""
        if date != None:
            if type(date) == "string":
                self.date = datetime.datetime.strptime(date,"%m-%d-%Y")
            elif type(date) == "datetime":
                self.date = date
            else:
                print "Supplied date not valid"
        for line in header:
            if line.find("Date") > -1:
                datestr = line.split(":")[-1].lstrip(" ")
                date = datetime.datetime.strptime(datestr,"%m-%d-%Y")
                if self.date == None:
                    self.date = date
                    self.dateString = datetime.datetime.strftime(self.date, "%m-%d-%Y")
            else:
                print "Header: " + line
        for line in events:    
            print line
            event = line.split(",")
            comment = event[0]
            start =  datetime.datetime.strptime(self.dateString + " " + event[1], "%m-%d-%Y %H:%M:%S")
            if event[2].find(":") > -1:
                end = datetime.datetime.strptime(self.dateString + " " + event[2], "%m-%d-%Y %H:%M:%S")
            else:
                end = None
            print str(comment) + " | " + str(start.strftime("%m-%d-%Y %H:%M:%S")) + " | " + str(end.strftime("%m-%d-%Y %H:%M:%S"))
            self.events.append(Event(comment=comment,startTime=start,endTime=end))
'''