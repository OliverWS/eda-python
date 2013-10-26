import os
import sys
from qLogFile import qLogFile
from zipfile import *
from xml.dom import minidom
from xml.dom.minidom import getDOMImplementation
from xml.sax.saxutils import unescape as unescape_xml
from events import *
import exceptions
import logging as log

def unescape(s):
	if s.startswith("<![CDATA["):
		s = s.replace("<![CDATA[","")
	if s.endswith("]]>"):
		s = s.replace("]]>", "")
	
	return unescape_xml(s)

class qPro:
    
    colors = ["0x2CC4FF","0xF52D7E","0xA3D62F","0xFF9800","0x11A985","0x336292","0x3F3F3F","0x660066","0xF2EC19","0x8F5B0B","0xA62C0F"]
    graphMode = "historyMode"
    def __init__(self, path=None):
        self.files = []
        self.timelineEvents = []
        self.events = self.timelineEvents
        self.version = 1.2
        self.zipfile = None
        self.metadata = dict();
        self.filepath = path
        if path != None:
            self.filename = os.path.split(path)[-1]
        if path != None:
            self.importProject(path)
    
    @property
    def startTime(self):
        return min(map(lambda x: x.startTime, self.files)) 
    @property
    def endTime(self):
        return max(map(lambda x: x.endTime, self.files)) 

    
    def importProject(self,path):
        projectFile =  ZipFile(path, 'r')
        manifestFile = projectFile.open("project.xml")
        manifest = minidom.parse(manifestFile)
        self.version = float(manifest.getElementsByTagName("Project")[0].getAttribute("version"))
        if self.version < 1.2:
            log.exception("Problem loading %s: Project file version is too old (%0.2f). Only project files of version 1.2 and above are supported"%(path,self.version))
        for datasetNode in manifest.getElementsByTagName("Dataset"):
            if len(datasetNode.getElementsByTagName("Csv")) > 0:
                log.info(datasetNode)
                fileNode = datasetNode.getElementsByTagName("Csv")[0]
                filename = str(fileNode.firstChild.wholeText)
            else:
                fileNode = datasetNode.getElementsByTagName("File")[0]
                filename = str(fileNode.firstChild.wholeText)
                if ".eda" in filename:
                    pass
                else:
                    filename = ""
                    log.error("Could not load dataset: %s"%(filename))
            dataset = None
            try:
                file = projectFile.open(filename)
                log.info("%s"%(str(file)))
                dataset = qLogFile(file, filename=filename.replace(".csv",".eda"), verbose=True)
                dataset.events = []
            except exceptions.Exception as e:
                log.exception("Problem opening %s in project file %s: %s"%(filename,path, str(e)))
            for eventNode in datasetNode.getElementsByTagName("Event"):
                start = int(eventNode.getAttribute("offset"))
                end = int(eventNode.getAttribute("partner")) if eventNode.hasAttribute("partner") else None
                if end != None and start > end:
                    start,end = end,start
                start = (start/1000.0)
                end = end/1000.0 if end != None else None
                try:
                    note = unescape(str(eventNode.getElementsByTagName("Notes")[0].firstChild.wholeText))
                except:
                    note = ""
                isDeviceEvent = True if str(eventNode.getAttribute("device")) == "true" else False
                if ( not end in map(lambda x: x.start, dataset.events) and not end in map(lambda x: x.end, dataset.events) )or end == None:
                    event = Event(start,end=end,note=note,isDeviceEvent=isDeviceEvent, file=dataset)
                    if dataset != None:
                        dataset.events.append(event)
            if dataset != None:
                self.files.append(dataset)
        for eventsNode in manifest.getElementsByTagName("Timeline")[0].getElementsByTagName("Events"):
            for eventNode in eventsNode.getElementsByTagName("Event"):
                start = int(eventNode.getAttribute("offset"))
                end = int(eventNode.getAttribute("partner")) if eventNode.hasAttribute("partner") else None
                start = start/1000.0
                end = (end/1000.0) if end != None else None
                try:
                    note = unescape(str(eventNode.getElementsByTagName("Notes")[0].firstChild.wholeText))
                except:
                    note = ""
                isDeviceEvent = False
                if not end in map(lambda x: x.start, self.timelineEvents) or end == None:
                    event = Event(start,end=end,note=note,isDeviceEvent=isDeviceEvent, file=None)
                    self.timelineEvents.append(event)


        projectFile.close()
        
    def xmlForDataset(self, dataset):
        defaultDatasetNode = '''<Dataset accel="true" alignmentOffset="0" alignmentStart="-1" color="%s" visible="true"><File>%s</File><Events></Events></Dataset>'''%(self.colors[self.files.index(dataset)%len(self.colors)], dataset.filename)
        datasetNode = minidom.parseString(defaultDatasetNode)
        eventsNode = datasetNode.getElementsByTagName("Events")[0]
        if hasattr(dataset,"events"):
            for event in dataset.events:
                if event.isRange():
                    for e in event.toxml():
                        eventsNode.appendChild(e)
                        log.info(str(e))
                else:
                    eventsNode.appendChild(event.toxml())
        return datasetNode.firstChild
    
    
    def extractFiles(self,path=None):
        if path == None:
            path = os.path.join(os.path.split(self.filepath)[0],self.filename.replace(".qpro",""))
            if not os.path.exists(path):
                os.makedirs(path)
        for dataset in self.files:
            dataset.save(os.path.join(path,dataset.filename))
    
    
    def saveProject(self, path, qlpreview=False):
        projectFile =  ZipFile(path, 'w')
        defaultManifest = '''<Project version="%0.1f"><GraphMode>%s</GraphMode><Datasets></Datasets><Timeline><Events/></Timeline></Project>'''%(self.version, self.graphMode)
        manifest = minidom.parseString(defaultManifest)
        datasetsNode = manifest.getElementsByTagName("Datasets")[0]
        n = 0
        for dataset in self.files:
            datasetsNode.appendChild(self.xmlForDataset(dataset))
            projectFile.writestr(dataset.filename, dataset.toString(), ZIP_DEFLATED)

        '''
            if qlpreview:
                import matplotlib
                matplotlib.use('Agg')
                from matplotlib import pyplot as plt
                import urllib
                import matplotlib.ticker as ticker
                from matplotlib.ticker import MultipleLocator, FormatStrFormatter
                fig = plt.figure()
                plt.gcf().set_size_inches(8,4)
                eda = np.array(dataset.EDA())
                plt.plot(eda, color=self.colors[n].replace("0x","#"))
                n = n + 1
                plt.xlim(0,len(eda))
                plt.ylim(0, np.max(eda))
                fig.axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(dataset.formatTime))
                fig.axes[0].xaxis.set_major_locator(MultipleLocator(len(eda)/6)) 
        if qlpreview:
            plt.savefig("/tmp/temp.png")
            tmp = open("/tmp/temp.png", "rb")
            imageData = tmp.read()
            tmp.close()
            projectFile.writestr(os.path.join("QuickLook","Preview.png"), imageData, ZIP_DEFLATED)
            projectFile.writestr(os.path.join("QuickLook","Thumbnail.png"), imageData, ZIP_DEFLATED)
'''
        #Now do the Timeline Events
        eventsNode = manifest.getElementsByTagName("Timeline")[0].getElementsByTagName("Events")[0]
        for event in self.timelineEvents:
            if event.isRange():
                for e in event.toxml():
                    eventsNode.appendChild(e)
            else:
                eventsNode.appendChild(event.toxml())
        projectFile.writestr("project.xml", manifest.firstChild.toxml(), ZIP_DEFLATED)
        projectFile.close()
                
        
        
        
        
        
        
        
       
