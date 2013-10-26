Usage:
from edatoolkit import qLogFile
edaObject = qLogFile("/path/to/eda/file") #e.g. qLogFile("test.eda")

#The below methods return data as lists (for historical reasons)
eda = edaObject.EDA()
x = edaObject.ACC_X()
y = edaObject.ACC_Y()
z = edaObject.ACC_Z()
temperature = edaObject.Temperature()

#Or, if you prefer numpy arrays you can use the property accessors
eda = edaObject.eda
x = edaObject.acc_x
y = edaObject.acc_y
z = edaObject.acc_z
temperature = edaObject.temperature
timestamps = edaObject.timestamps # This delivers a numpy array of timestamps corresponding to each sample (great for doing np.plot(timestamps,eda) )
#To save an edafile to a file (always uses .csv edaformat)
edaObject.save("path/to/save/file")

#Other useful methods
edaObject.timeForOffset(idx) # Returns a python time object representing time at row in data file
edaObject.offsetForTime(datetime) # Returns index into array for python datetime object (make sure it has tzinfo)
edaObject.startTime #Returns start time of file as datetime object
edaObject.endTime #Returns end time of file as datetime object
edaObject.setEDA(newEDAList) #Overwrites EDA channel with your new data (must be same size!)
