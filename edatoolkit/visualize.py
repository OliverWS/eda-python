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

from toolkit import *
from colorsys import *
import exceptions
import matplotlib
from math import *
from matplotlib import pyplot as plt
import matplotlib.units as units
import matplotlib.dates as dates
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import qLogFile
from _utils import *

import ais.logger
afflogger = ais.logger.get_affdex_logger()


def radial(edafile,rightEDA = None,hoursPerArc = 12, datatype="eda", width=1024, height=1024, path=None, scheme="red",grad=None, showMotion=True):
	from PIL import Image as PImage
	#from affdex import utils as aff
	import drawing
	import scipy
	import math
	import qDataProcessor
	import sys
	import datetime as dt
	if not grad == None:
		scheme = "gradient"
		if grad.__class__.__name__ == "Gradient":
			gradient = grad
		else:
			gradient = Gradient(grad)
	else:
		gradient = Gradient([(0,0,0,255),(255,0,0,255),(255,255,0,255)])
	
	def colorForPixel(val, scheme="red", grad=gradient):
		if scheme == "red":
			r,g,b,a = grad.getColor(val*512)
		elif scheme == "blue":
			r,g,b,a = grad.getColor(val*512)
		return (r,g,b,a)
	
	if path == None:
		path = os.path.join(os.path.abspath(os.curdir), edafile.filename.replace(".eda",".png"))
	elif path.find("/") == -1:
		path = os.path.join(os.path.abspath(os.curdir), path)

	r = (width*0.9)/2.0 #radius, with 10% headroom
	rInner = r*0.75
	w = math.floor(math.pi*2.0*r)
	wInner = math.floor(math.pi*2.0*rInner)
	k = len(edafile.data)/w
	kInner =  len(edafile.data)/wInner
	if datatype == "eda":
		edafile.setEDA(scipy.ndimage.gaussian_filter1d(edafile.EDA(), k))
		edafile = qDataProcessor.normalizeData([edafile])[0]
		data = normaliseVector(np.array(scipy.ndimage.filters.gaussian_filter1d(edafile.EDA(), k)), timeResolution=w)
	elif datatype == "ppm":
		edafile = qDataProcessor.process([edafile], ["dilate","normalize","peaks","ppm"])[0]
		data = normaliseVector(np.array(edafile.ppm), timeResolution=w)
		data = data/np.max(data)
		if cutout:
			edafile.setEDA(scipy.ndimage.gaussian_filter1d(edafile.EDA(), k))
			edafile = qDataProcessor.normalizeData([edafile])[0]
			eda = normaliseVector(np.array(edafile.EDA()), timeResolution=w)
	
	if showMotion == True:
		accx = np.array(edafile.ACC_X())
		accy = np.array(edafile.ACC_Y())
		accz = np.array(edafile.ACC_Z())
		
		accCombo = np.sqrt(accx*accx + accy*accy + accz*accz)
		sr = int(edafile.sampleRate)
		#for i in xrange(0,len(accx),sr):
		#	accVar.append(np.std(accCombo[i:i+sr]))
		accCombo = (np.array(accCombo) - np.nanmin(accCombo))/np.nanmax(accCombo)
		accVar = scipy.ndimage.filters.gaussian_filter1d(scipy.ndimage.filters.maximum_filter1d(accCombo, int(kInner)), int(kInner))
		acc = (np.array(accVar) - np.nanmin(accVar))/np.nanmax(accVar)
		acc = normaliseVector(acc, timeResolution=wInner)
		accGradient = loadGradient("/Users/oliverws/modis_images/Fiji.grad")#Gradient([(0,0,0,255),(22,253,247,255)]) #
	elif rightEDA != None:
		rightEDA.setEDA(scipy.ndimage.gaussian_filter1d(rightEDA.EDA(), k))
		rightEDA = qDataProcessor.normalizeData([rightEDA])[0]
		rEDA = normaliseVector(np.array(scipy.ndimage.filters.gaussian_filter1d(rightEDA.EDA(), k)), timeResolution=w)
		innerGradient = gradient#loadGradient("/Users/oliverws/modis_images/Fiji.grad")#Gradient([(0,0,0,255),(22,253,247,255)]) #


		
	# Load the source file
	img = drawing.Context(width=width,height=height, background=(0,0,0,255))


	def time2deg(t, hours=hoursPerArc):
		offset = 0#-90.0 if hours <= 12 else 45.0
		seconds = (t - t.replace(hour=0,minute=0,second=0)).seconds
		fraction = (seconds/(hours*60.0*60.0))
		deg = fraction*360.0 + offset
		return deg
	fontPath = "/Users/oliverws/Fonts/OpenSans-Light.ttf"
	fontSize = int(math.floor(img.width/50))
	headerFontSize = int(math.floor(img.width/30))
	# Loop through image data
	for n in xrange(0,int(w)):
		x = (edafile.endTime - edafile.startTime).seconds/w
		offset = -90.0 if hoursPerArc <= 12 else -90.0
		arcStart = time2deg(edafile.startTime + dt.timedelta(seconds=x*n)) + offset
		arcEnd = time2deg(edafile.startTime + dt.timedelta(seconds=x*(n+1))) + offset
		img.arc(img.width/2,img.height/2,r,start=arcStart,end=arcEnd, stroke=(0,0,0,0), thickness=0.0, fill=colorForPixel(data[n]))
	img.arc(img.width/2,img.height/2,r*0.75, stroke=(0,0,0,0), thickness=0.0, fill=(0,0,0,255))
	
	if showMotion:
		for n in xrange(0,int(wInner)):
			x = (edafile.endTime - edafile.startTime).seconds/wInner
			arcStart = time2deg(edafile.startTime + dt.timedelta(seconds=x*n)) + offset
			arcEnd = time2deg(edafile.startTime + dt.timedelta(seconds=x*(n+1))) + offset
			img.arc(img.width/2,img.height/2,r*0.75,start=arcStart,end=arcEnd, stroke=(0,0,0,0), thickness=0.0, fill=colorForPixel(acc[n], scheme="blue", grad=accGradient))
	img.arc(img.width/2,img.height/2,r*0.625, stroke=(0,0,0,0), thickness=0.0, fill=(0,0,0,255))
	
	if rightEDA != None:
		for n in xrange(0,int(w)):
			x = (edafile.endTime - rightEDA.startTime).seconds/w
			arcStart = time2deg(rightEDA.startTime + dt.timedelta(seconds=x*n))
			arcEnd = time2deg(edafile.startTime + dt.timedelta(seconds=x*(n+1)))
			img.arc(img.width/2,img.height/2,r*0.75,start=arcStart,end=arcEnd, stroke=(0,0,0,0), thickness=0.0, fill=colorForPixel(rEDA[n], scheme="blue", grad=innerGradient))
	img.arc(img.width/2,img.height/2,r*0.625, stroke=(0,0,0,0), thickness=0.0, fill=(0,0,0,255))

	
	for n in xrange(-90, 270, 360/hoursPerArc):
		startx,starty = (img.width/2,img.height/2)
		y = np.sin(2*np.pi*(n)/360.0) * r*0.6 
		x = np.sqrt((r*0.6)*(r*0.6) - y*y) * (-1.0 if (n < 90 or n > 270) else 1.0)
		endx,endy = x + img.width/2, y + img.height/2
		afflogger.info("X = %d | Y = %d"%(x,y))
		img.line(startx,starty,endx,endy, stroke=(255,255,255,255), thickness=2.0)		

	
	img.arc(img.width/2,img.height/2,r*0.35, stroke=(0,0,0,0), thickness=0.0, fill=(0,0,0,255))
	'''
	for n in xrange(-90, 270, 360/hoursPerArc):
		if hoursPerArc <= 12:
			t = (hoursPerArc - (n +90)/(360/hoursPerArc))
			time = "%d %s"%(t, "AM" if t < 12 and t > (edafile.endTime.hour - 12) else "PM" )
		elif hoursPerArc >= 24:
			t = (hoursPerArc - (n +90)/(360/hoursPerArc))
			time = "%d"%(t)
		y = np.sin(2*np.pi*(n)/360.0) * r*0.25
		x = np.sqrt((r*0.25)*(r*0.25) - y*y) * (-1.0 if (n < 90 or n > 270) else 1.0)
		endx,endy = x + img.width/2 - math.floor(0.25*fontSize*len(time)), y + img.height/2 - math.floor(0.75*fontSize)
		img.text(time, fontPath, endx,endy, stroke=(255,255,255,255), size=fontSize)
	'''
	startHour = edafile.startTime.hour if hoursPerArc <= 12 else 0
	
	for z in xrange(edafile.startTime.hour,startHour+ hoursPerArc):
		t = edafile.startTime.replace(hour=z,minute=0,second=0)
		n = (time2deg(t, hours=hoursPerArc) + offset)
		n = 360 + n if n < 0 else n
		n = n - 360 if n > 360 else n
		time = t.strftime("%I%p" if hoursPerArc<=12 else "%H")
		radii = r*0.28
		y = np.sin(2*np.pi*(n)/360.0) * radii
		x = np.sqrt(radii*radii - y*y) * (1.0 if (n < 90 or n > 270) else -1.0)
		afflogger.info("N=%f X=%f Y=%f Hour=%d"%(n,x,y,z))
		endx,endy = x + img.width/2 - math.floor(0.25*fontSize*len(time)), y + img.height/2 - math.floor(0.75*fontSize)
		img.text(time, fontPath, endx,endy, stroke=(255,255,255,255), size=fontSize)

	
	
	headerText = "%s"%(edafile.startTime.strftime("%A, %B %d %Y"))
	img.text(headerText, fontPath, startx - math.ceil(0.25*len(headerText)*headerFontSize),math.floor(r/8.0 - headerFontSize*1.5), stroke=(255,255,255,255), size=headerFontSize)

	img.image.save(path)
	return img




def heatmap(edafile, datatype="eda", width=None, height=64, cutout=False, path=None, scheme="red",grad=None, pixelsPerMinute=1):
	from PIL import Image as PImage
	#from affdex import utils as aff
	import scipy
	import qDataProcessor
	import sys
	import os
	gradient = None
	if not grad == None:
		scheme = "gradient"
		if grad.__class__.__name__ == "Gradient":
			gradient = grad
		else:
			gradient = Gradient(grad)
	
	def colorForPixel(val, scheme="red"):
		n = int(val/(1.0/512.0))
		a = 255
		if scheme == "red":
			r = n if n < 255 else 255
			g = n - 255 if n > 255 else 0
			b = 0
		elif scheme == "blue":
			b = n if n < 255 else 255
			g = n - 255 if n > 255 else 0
			r = 0
		elif scheme == "purple":
			b = n if n < 255 else 255
			g = n - 255 if n > 255 else 0
			r = n if n < 255 else 255
		elif scheme == "green":
			g = n if n < 255 else 255
			b = int((n - 255)*0.8) if n > 255 else 0
			r = int((n - 255)*0.8) if n > 255 else 0
		elif scheme == "gradient":
			r,g,b,a = gradient.getColor(n)
		return (r,g,b,a)
	
	if path == None:
		path = os.path.join(os.path.abspath(os.curdir), edafile.filename.replace(".eda",".png"))
	elif path.find("/") == -1:
		path = os.path.join(os.path.abspath(os.curdir), path)

	h = height*4 if cutout else height
	w = int(floor((len(edafile.data)/int(edafile.sampleRate*60))*pixelsPerMinute)) if width == None else width
	k = len(edafile.data)/w
	if datatype == "eda":
		edafile.setEDA(scipy.ndimage.gaussian_filter1d(edafile.EDA(), k))
		edafile = qDataProcessor.normalizeData([edafile])[0]
		data = normaliseVector(np.array(edafile.EDA()), timeResolution=w)
	elif datatype == "ppm":
		edafile = qDataProcessor.process([edafile], ["dilate","normalize","peaks","ppm"])[0]
		data = normaliseVector(np.array(edafile.ppm), timeResolution=w)
		data = data/np.max(data)
		if cutout:
			edafile.setEDA(scipy.ndimage.gaussian_filter1d(edafile.EDA(), k))
			edafile = qDataProcessor.normalizeData([edafile])[0]
			eda = normaliseVector(np.array(edafile.EDA()), timeResolution=w)

	# Load the source file
	img = PImage.new("RGBA",(w,h))
	pixdata = img.load()

	# Loop through image data
	for x in xrange(img.size[0]):
		for y in xrange(img.size[1]):
			if cutout:
				if (h - int(eda[x]*h)) < y:
					pixdata[x,y] = colorForPixel(data[x],scheme=scheme)
				else:
					pixdata[x,y] = (0,0,0,0)
			else:
				pixdata[x,y] = colorForPixel(data[x],scheme=scheme)
	img.save(path)
	return img
	



def plot(log, width=11, height=11, save=False):
	EDA = log.EDA()
	plt.gcf().set_size_inches(width,height)
	fig = plt.figure(1)
	plt.subplot(511)
	plt.plot(log.EDA(), color="#2CC4FF")
	plt.ylabel("EDA ($\mu$S)")
	plt.subplot(512)
 	plt.plot(log.ACC_X(), color='r')
 	plt.ylabel("X Accleration (g)")
 	plt.ylim([-6.0, 6.0])
 	plt.subplot(513)
 	plt.plot(log.ACC_Y(), color='g')
 	plt.ylabel("Y Accleration (g)")
 	plt.ylim([-6.0, 6.0])
 	plt.subplot(514)
 	plt.plot(log.ACC_Z(), color='b')
 	plt.ylim([-6.0, 6.0])
 	plt.ylabel("Z Accleration (g)")
 	plt.subplot(515)
 	plt.plot(log.Temperature(), color='#2CC4FF')
 	plt.ylim([0.0, 42.0])
	plt.ylabel("Temperature ($^{\circ}$C)")
	fig.subplots_adjust(hspace=0.25)
	#fig.autofmt_xdate()
	for sp in fig.axes:
		sp.xaxis.set_major_formatter(ticker.FuncFormatter(log.formatTime))
		sp.xaxis.set_major_locator(MultipleLocator(len(EDA)/6))
		sp.axis.im_self.set_xlim([0, len(EDA)])
	if not save == False:
		afflogger.info("Saving figure for " + log.filename)
		if os.path.isdir(save):
			path = os.path.join(sys.argv[1], log.filename.rstrip(".eda"))
		else:
			path = log.filename.rstrip(".eda")
		plt.savefig(path, dpi=200)
	else:
		plt.show()
	plt.close()

def plotACC(log):
	x = log.ACC_X()
	y = log.ACC_Y()
	z = log.ACC_Z()
	matplotlib.pyplot.gcf().set_size_inches(11,8.5)
	dx = np.diff(x)
	dy = np.diff(y)
	dz = np.diff(z)
	fig = plt.figure(1)
	plt.subplot(311)
	#plt.title(log.filename)
	spm = 60*log.sampleRate
	step = log.sampleRate*1
	var = []
	for d in range(0, len(dx)-spm, step):
		var.append(np.std(dx[d : d + spm]))
	track = np.arange(0, len(dx)-1-spm, step)
	plt.subplot(311)
	plt.plot(x, color="r")
	plt.xlim(0, len(x) - 1)
	plt.ylabel("X Accleration (g)")
	plt.subplot(312)
	plt.plot(dx, color="r")
 	plt.ylabel("$\Delta$X Accleration")
 	plt.xlim(0, len(dx) - 1)
	plt.subplot(313)
	plt.bar(track, var, step, color="purple")
	plt.ylabel("Jerkiness Score (per Minute)")
	plt.ylim(0, 1.0)
	plt.xlim(0, len(dx)-1-spm)
	fig.subplots_adjust(hspace=0.4)
	for sp in fig.axes:
		sp.xaxis.set_major_formatter(ticker.FuncFormatter(log.formatTime))
		sp.xaxis.set_major_locator(MultipleLocator(len(x)/6))
	if len(sys.argv) > 3 and (sys.argv[3] == "save" or os.path.isdir(sys.argv[1])):
		afflogger.info("Saving figure for " + log.filename)
		if os.path.isdir(sys.argv[1]):
			path = os.path.join(sys.argv[1], log.filename.rstrip(".eda"))
		else:
			path = log.filename.rstrip(".eda")
		plt.savefig(path, dpi=200)
	else:
		plt.show()
	plt.clf()
	plt.close()


def toMovieFile(log, window=60, fps=8.0):
	dir = "tmp"+log.filename.rstrip(".eda")
	sr = int(log.sampleRate)
	EDA = np.array(log.EDA())
	if os.path.isdir(dir):
		shutil.rmtree(dir)
	os.mkdir(dir)
	index = 0
	window = int(window*log.sampleRate)
	step = int(math.ceil(sr*(1.0/fps))) #Sliding window increments: 1 second
	afflogger.info("Sliding window of " + str(window/sr) + " seconds in increments of " + str(float(step)/float(sr)) + " seconds for FPS=" + str(fps))
	maxCompleteIndex = (len(EDA) - window) - ((len(EDA) - window)%step)
	formatString = "{0:0" + str(len(str(len(EDA)))) + "d}" 
	for n in range(0, maxCompleteIndex, step):
		path = os.path.join(dir, formatString.format(index)+".png")
		signal = EDA #[n : n + window]
		plt.clf()
		fig = plt.figure(1)
		matplotlib.pyplot.gcf().set_size_inches(8,2)
		#Change what you plot in here to change what the movie shows!
		ax = plt.plot(signal, color="#2CC4FF")
		plt.ylabel("EDA ($\mu$S)")
		plt.ylim(EDA.min()*0.9, EDA.max()*1.1)
		plt.axis([n, n+window,0, EDA.max()*1.1])
		fig.axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(log.formatTime))
		#fig.axes[0].xaxis.set_major_locator(MultipleLocator(len(signal)/(4*window)))
		plt.xlim(n, n + window)
		#Don't change below here...
		plt.savefig(path)
		index = index + 1
	generateMovieFromImages(dir, log.filename.rstrip(".eda"), fps)

def toMovieFileMovingCursor(log):
	dir = "tmp2"+log.filename.rstrip(".eda")
	sr = int(log.sampleRate)
	height = 100
	width = 800
	dpi = 110
	EDA = np.array(lowpassSignal(medfilter(log).EDA(), sr, sr/4))
	EDA = EDA[125000 : 125000 + sr*60]
	if os.path.isdir(dir):
		shutil.rmtree(dir)
	os.mkdir(dir)
	index = 0
	if len(sys.argv) > 4 and sys.argv[4].startswith("window="):
		window = int(sys.argv[4].split("=")[1])*sr
	else:
		window = sr*60*5
	if len(sys.argv) > 3 and sys.argv[3].startswith("fps="):
		fps = float(sys.argv[3].split("=")[1])
	else:
		fps = 8.0
	step = int(math.ceil(float(sr)*(1.0/fps))) #Sliding window increments: 1/fps second
	afflogger.info("Sliding window of " + str(window/sr) + " seconds in increments of " + str(step/sr) + " seconds for FPS=" + str(fps))
	maxCompleteIndex = (len(EDA) - 1) - ((len(EDA))%step)
	formatString = "{0:0" + str(len(str(len(EDA)))) + "d}" 
	for n in range(0, maxCompleteIndex, step):
		#print "Creating frame #" + str(n) + " of " + str(maxCompleteIndex/step)
		fn = formatString.format(index) + ".png"
		path = os.path.join(dir, fn)
		signal = EDA
		plt.clf()
		fig = plt.figure(1)
		matplotlib.pyplot.gcf().set_size_inches(16,2)
		#Change what you plot in here to change what the movie shows!
		plt.plot(signal)
		plt.ylabel("EDA ($\mu$S)")
		plt.axvline(n, color="red", linewidth=1)
		plt.ylim(EDA.min()*0.9, EDA.max()*1.1)
		#plt.axis([n, n+window,0, EDA.max()])
		plt.xlim(0, len(signal) - 1)
		#Don't change below here...
		plt.savefig(path)
		index = index + 1
		if n%(step*1000) == 0:
			time.sleep(1)
	generateMovieFromImages(dir, log.filename.rstrip(".eda"), fps)


def generateMovieFromImages(path, filename, fps):
	commandStr = "mencoder mf://" + os.path.join(path, "*.png") + " -mf type=png:fps=" + str(fps) + " -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o " + str(filename) + ".avi"
	afflogger.info("Generating movie file...")
	os.system(commandStr)
	afflogger.info("Cleaning up temporary files")
	#shutil.rmtree(path)
	afflogger.info("Done")


def spectogram(log):
	NFFT = 1024
	fs = int(log.sampleRate)
	EDA = lowpassSignal(log.ACC_X(), fs, fs/4)
	threshold = 0.05
	peaks = []
	fig = figure(1)
	plt.subplot(211)
	plt.title(log.filename)
	plt.specgram(EDA, NFFT=NFFT, Fs=fs, noverlap=512, scale_by_freq=False)
	plt.ylabel("Frequency (Hz)")
	plt.ylim(0, fs/4)
	plt.xlim(0, 5574)
	plt.subplot(212)
	plt.plot(log.EDA())
	plt.ylabel("EDA ($\mu$S)")
	plt.xlabel("Time")
	fig.axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(log.formatTime))
	fig.axes[-1].xaxis.set_major_locator(MultipleLocator(len(EDA)/6))

	if len(sys.argv) > 3 and (sys.argv[3] == "save" or sys.argv[3] == "csv"):
		afflogger.info("Saving figure")
		title = log.filename.rstrip('.eda') + "_spectogram"
		if os.path.isdir(sys.argv[1]):
			filename = os.path.join(sys.argv[1], title)
		else:
			filename = title
		plt.savefig(filename, dpi=200)
	else:
		plt.show()







def diagram(logfile, path=None, format=".png", peakThreshold=0.0, fontSize=None, lineLength=50, subset=None,lineWidth=1.0, transparent=True):
	from matplotlib import pyplot as plt
	import matplotlib.ticker as ticker
	import os
	from matplotlib.ticker import MultipleLocator, FormatStrFormatter
	plt.clf()
	filename = logfile.filename.replace(".eda",format)
	fig = plt.figure()
	if subset == None:
		subset = (0,len(logfile.data))
	eda = logfile.smoothEDA[subset[0]:subset[1]]
	w = int(len(eda)/float(logfile.sampleRate))*150/72 #np.amin(np.diff([x for x in logfile.peaks if x >subset[0] and x < subset[1]]))*150
	
	h = w/10#np.amin(map(lambda x : x["rise"], logfile.peakList))*100
	afflogger.info("Width: %d | Height: %d"%(w,h))
	plt.gcf().set_size_inches(w,h)
	plt.plot(eda, color="#2CC4FF", lw=lineWidth*1.5)
	ax = fig.axes[0]
	
	fontSize = (h*72)/90 if fontSize == None else fontSize
	
	for p in [x for x in logfile.peakList if (x["rise"] > peakThreshold and x["start"] > subset[0] and x["end"] < subset[1])]:
		x = p["index"] - subset[0]
		start = p["start"] - subset[0]
		end = p["end"] - subset[0]
		y = eda[x]
		m = (eda[x] - eda[start])/2 + eda[start]
		m2 = (end - x)/2 + x
		s = "%0.2f$\mu$S\nPeak Height"%(p["rise"])
		#plt.text(x,y,s,horizontalalignment='center', verticalalignment='center', size=5, color="white", weight="bold")
		
		plt.fill_between(np.arange(start,end), eda[np.arange(start,end)], p["boundary"], color="#B7E2F3")
		plt.annotate("AUC=%0.2f$\mu$S/sec"%(p["AUC"]),(x,eda[x]),(x+(end-x)/3,eda[x] + 0.03),arrowprops=dict(arrowstyle= "-",connectionstyle="arc,angleA=0,armA=20,angleB=-90,armB=15,rad=7"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")

		
		#plt.text(x,eda[start] + (eda[x]-eda[start])/2.0,"%0.2f$\mu$S/s"%(p["AUC"]),horizontalalignment='center', verticalalignment='center', size=fontSize, color="white", weight="bold")

		plt.plot([start-lineLength,x],[y, y],  color="black", lw=lineWidth)
		plt.plot([start-lineLength,start],[eda[start], eda[start]],  color="black",  lw=lineWidth)
		plt.plot([x,x],[y, eda[start] -0.05],  color="black",  lw=lineWidth)
		plt.plot([end,end],[eda[end], eda[start]- 0.1],  color="black",  lw=lineWidth)
		plt.plot([start,start],[eda[start], eda[start]- 0.1],  color="black",  lw=lineWidth)
		
		plt.plot([end,x],[y, y],  color="black", lw=lineWidth)
		plt.plot([end,end],[eda[end], eda[end]],  color="black",  lw=lineWidth)
 		fall = "%0.2f$\mu$S\nFall"%(p["fall"])
 		plt.annotate(fall,(end,eda[x]),(end,eda[x] - (eda[x] - eda[end])/2.0),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")
 		plt.annotate(fall,(end,eda[end]),(end,eda[x] - (eda[x] - eda[end])/2.0),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")

		#plt.plot([start,x],[eda[start], eda[x]],  color="black",  lw=lineWidth)

		
		
		
		#plt.arrow(start-lineLength,  + 0.25, 0, (eda[x]-eda[start])/2,length_includes_head=True, head_width=0.5, head_length=1, width=0.001)
 		plt.annotate(s,(start-lineLength,eda[start]),(start-lineLength,eda[x] - (eda[x] - eda[start])/2.0),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")
 		plt.annotate(s,(start-lineLength,eda[x]),(start-lineLength,eda[x] - (eda[x] - eda[start])/2.0),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")
 		
 		t2 = "%0.1fs\nFall Time"%((end - x)/logfile.sampleRate)
 		t1 = "%0.1fs\nRise Time"%((x - start)/logfile.sampleRate)
 		t3 = "%0.1fs\nDuration"%((end - start)/logfile.sampleRate)
 		plt.annotate(t1,(start,eda[start] -0.05),(start + (x-start)/2,eda[start] - 0.05),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")
 		plt.annotate(t1,(x,eda[start] -0.05),(start + (x-start)/2,eda[start] - 0.05),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")
 		plt.annotate(t2,(end,eda[start] -0.05),(x + (end-x)/2,eda[start] - 0.05),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")
 		plt.annotate(t2,(x,eda[start] -0.05),(x + (end-x)/2,eda[start] - 0.05),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")

 		#Duration
 		plt.annotate(t3,(end,eda[start] -0.1),(start + (end-start)/2 ,eda[start] - 0.1),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")
 		plt.annotate(t3,(start,eda[start] -0.1),(start + (end-start)/2,eda[start] - 0.1),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")
 		#plt.annotate(s,(start-lineLength,eda[x]),(start-lineLength,eda[x] - (eda[x] - eda[start])/2.0),arrowprops=dict(arrowstyle= "->"),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")

 		#plt.annotate("",(2500,0.0),(5000,0.5),arrowprops=dict(arrowstyle= "<->"))

		#plt.text(m2,eda[end]-h/40.0,"%0.1fs"%((end - x)/logfile.sampleRate),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")
		#plt.text(start + (x-start)/2,eda[end]-h/40.0,"%0.1fs"%(p["riseTime"]),horizontalalignment='center', verticalalignment='center', size=fontSize, color="black", weight="bold")

		#plt.text(start-100,m,s,s)

		#plt.axhline(y=eda[end], xmin=x-10, xmax=x+5, transform=ax.transData)
		#plt.annotate(s,(x,y),(x,y+0.05), color="black")
	plt.xlim(0,len(eda))
	plt.ylim(np.amin(eda)*0.9, np.max(eda)*1.1)
	#fig.axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(logfile.formatTime))
	#fig.axes[0].xaxis.set_major_locator(MultipleLocator(len(eda)/6))
	if path != None:
		plt.savefig(os.path.join(path,filename), bbox_inches='tight', dpi=72, transparent=transparent)
	else:
		plt.show()




def loadGradient(path):
	g = open(path, "rb").read()
	return eval(g)

class Gradient:
	def __init__(self, stops=None, resolution=512):
		import numpy as np
		if stops != None:
			r = map(lambda x : x[0], stops)
			g = map(lambda x : x[1], stops)
			b = map(lambda x : x[2], stops)
			a = map(lambda x : x[3], stops)
			from affdex import utils as aff
			self.r = aff.normaliseVector(np.array(r), timeResolution=resolution)
			self.g = aff.normaliseVector(np.array(g), timeResolution=resolution)
			self.b = aff.normaliseVector(np.array(b), timeResolution=resolution)
			self.a = aff.normaliseVector(np.array(a), timeResolution=resolution)
			self.resolution = resolution
			self.stops = stops

	def getColor(self, index):
		index = self.resolution-1 if index > self.resolution-1 else index
		try:
			return (int(self.r[index]),int(self.g[index]), int(self.b[index]),int(self.a[index]))
		except exceptions.Exception as e:
			afflogger.exception(e)
			afflogger.info(str(index))
			afflogger.info(str( int(self.r[index]),int(self.g[index]), int(self.b[index]),int(self.a[index])))

	def show(self, filename=None):
		from PIL import Image
		img = Image.new("RGBA",(self.resolution,150))
		pixdata = img.load()
		
		for y in xrange(0, img.size[1]):
			for x in xrange(0, img.size[0]):
				pixdata[x,y] = self.getColor(x)
		
		if filename != None:
			img.save(filename)
		else:
			img.show()

	def archive(self,filename):
		f = open(filename, "w")
		f.write("Gradient(%s)"%(repr(self.stops)))
		f.close()





class edaViz:
	
	def __init__(self, log):
		self.log = log
		self.sampleRate = log.sampleRate
		self.startTime = log.startTime
		self.endTime = log.endTime
		self.filename = log.filename
		self.inc = 60
		self.width = 24.0
		self.height = 2.0
		self.ppi = 300
	def viz(self, type="ppm", inc=None, h=0.53, s=0.89):
		if inc != None:
			self.inc = inc
		else:
			self.inc = ((self.endTime - self.startTime).seconds/(self.width*self.ppi))*3
			afflogger.info(str(self.inc))
		data = np.array(computePPM(self.log, increment=self.inc))
		max = data.max()
		min = data.min()
		fig = plt.figure(figsize=(self.width,self.height), dpi=self.ppi, frameon=False)
		for n in range(1, len(data)):
			r,g,b = self.mapColor(data[n], min, max,h=h,s=s)
			color = [r,g,b,1.0]
			plt.axvspan(n-0.5,n+0.5, linewidth=0.0, color=color)
		#plt.ylim(0, tonic.max())
		plt.xlim(0,len(data))
		ticks, labels = self.getTimeTicksAndLabels(data)
		plt.xticks(ticks, labels, rotation=45, fontsize=12, ha='right', color="white")
		fig.axes[-1].yaxis.set_ticks_position("none")
		fig.axes[-1].set_yticklabels([])
		plt.title(self.log.startTime.strftime("%B, %d %Y"), color="white")
		plt.savefig(self.log.filename.rstrip(".eda")+"_Viz.png", bbox_inches='tight', dpi=300, transparent=True)
		plt.clf()
	
	def mapColor(self, c, min, max, h=0.53, s=0.89):
		m = (float(c) - float(min))/(float(max) - float(min))
		return hsv_to_rgb(h, s, m)
	
	def formatTime(self, x, pos=None):
		t = x*self.inc
		return (self.startTime + datetime.timedelta(seconds=t)).strftime("%I:%M:%S %p")
	
	def getTimeTicksAndLabels(self, data):
		ticks = self.getHoursIndex()
		ticks.insert(0,0)
		ticks.append(len(data)-1)
		labels = map(self.formatTime, ticks)
		return (ticks, labels)
	
	def getHoursIndex(self):
		firstHour = self.startTime.hour+1
		lastHour = self.endTime.hour
		hours = []
		if self.endTime.day  == self.startTime.day:
			for n in range(firstHour, lastHour+1):
				hours.append(datetime.datetime(year = self.startTime.year, month = self.startTime.month, day = self.startTime.day, hour=n, minute=0, second=0))
			return map(lambda x : x/(self.inc*self.sampleRate), map(self.log.offsetForTime, hours))
			
# def timeFormat(x, pos=None):
# 	delta = 1.0/logfile.sampleRate
# 	t = x*inc
# 	return (logfile.startTime + datetime.timedelta(seconds=t)).strftime("%I:%M:%S %p")
# 
# logfile = qLogFile(sys.argv[1])
# inc = 60
# scaleFactor = 1
# if sys.argv[2] == "ppm":
# 	logfile.filename= logfile.filename.split(".")[0]+"_PPM"
# 	data = np.array(computePPM(logfile, increment=inc, threshold=0.025))
# 	viz(logfile, data)
# elif sys.argv[2] == "edv":
# 	logfile.filename= logfile.filename.split(".")[0]+"_EDV"
# 	peaks = findSCRs(logfile)
# 	sr = int(logfile.sampleRate)
# 	EDA = scipy.ndimage.filters.gaussian_filter1d(logfile.EDA(), sr, order=0)
# 	EDA1 = scipy.ndimage.filters.gaussian_filter1d(logfile.EDA(), sr, order=1)
# 	EDA2 = scipy.ndimage.filters.gaussian_filter1d(logfile.EDA(), sr, order=2)	
# 	isLocalMax = scipy.ndimage.maximum_filter1d( EDA, sr*2+1) == EDA
# 	isLocalMin = scipy.ndimage.minimum_filter1d( EDA, sr*2+1) == EDA
# 	maxIndex = np.nonzero( isLocalMax )[0]
# 	spm = int(logfile.sampleRate*60)
# 	ppm =[]
# 	for i in range(0, len(logfile.EDA())-spm-1,inc):
# 		ppm.append(isLocalMax[i:i+spm].sum())
# 	viz(logfile, np.array(ppm))