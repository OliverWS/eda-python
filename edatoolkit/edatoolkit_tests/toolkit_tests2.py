import os, sys
cmd_folder = os.path.dirname(os.path.abspath(__file__))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
lib_folder = os.path.dirname("../../")
if lib_folder not in sys.path:
    sys.path.insert(0, lib_folder)

import unittest
import edatoolkit
import edatoolkit.qLogFile as Log_File


class edatoolkitTest(unittest.TestCase):
    


    #def __len__(self):
#	return 0
  

    def setUp(self):
        
        self.baseFile1 = './resources/test_binary.eda'
        self.baseFile2 = './resources/test_text.eda'
        self.baseFile3 = './resources/test.eda'
        self.baseFile4 = './resources/test_headless.eda'
        self.baseFile5 = './resources/Faulty_Header.eda'
        self.eda1 = Log_File(self.baseFile1)
        self.eda2 = Log_File(self.baseFile2)
        self.eda3 = Log_File(self.baseFile3)
   

    def testReadBinaryFileAccAxis1(self):
        got_samples =len( self.eda1.ACC_X())
        desired_samples = 90030
        self.assertEqual(got_samples, desired_samples)

    def testReadBinaryFileAccAxis2(self):
        got_samples = len(self.eda1.ACC_Y())
        desired_samples = 90030
        self.assertEqual(got_samples, desired_samples)

    def testReadBinaryFileAccAxis3(self):
        got_samples =  len(self.eda1.ACC_Z())
        desired_samples = 90030
        self.assertEqual(got_samples, desired_samples)

    def testReadBinaryFileEDA(self):
        got_samples =  len(self.eda1.EDA())
        desired_samples = 90030
        self.assertEqual(got_samples, desired_samples)


    def testReadBinaryFiletemperature(self):
        got_samples =  len(self.eda1.Temperature())
        desired_samples = 90030
        self.assertEqual(got_samples, desired_samples)

    #def testvalidateSampleRate(self):
   	#got_header = self.eda3.sampleRate
        #desired_header = 2 or 4 or 8 or 16 or 32
        #self.assertEqual(got_header, desired_header)
    
    #def testvalidateSensorID(self):
   	#got_header = len(self.eda3.sensorID)
        #desired_header = 11
        #self.assertEqual(got_header, desired_header)
 
   
    #def testvalidateFileVersion(self):
       #	got_header = self.eda3.fileVersion
        #desired_header = 1.50
        #self.assertEqual(got_header, desired_header)

   # def testvalidateFirmwareVersion(self):
   	#got_header = self.eda3.firmwareVersion
        #desired_header = 1.50
       # self.assertEqual(got_header, desired_header)

    def testReadTextFileAccAxis1(self):
        got_samples =len( self.eda2.ACC_X())
        desired_samples = 90028
        self.assertEqual(got_samples, desired_samples)

    def testReadTextFileAccAxis2(self):
        got_samples = len(self.eda2.ACC_Y())
        desired_samples = 90028
        self.assertEqual(got_samples, desired_samples)

    def testReadTextFileAccAxis3(self):
        got_samples =  len(self.eda2.ACC_Z())
        desired_samples = 90028
        self.assertEqual(got_samples, desired_samples)

    def testReadTextFileEDA(self):
        got_samples =  len(self.eda2.EDA())
        desired_samples = 90028
        self.assertEqual(got_samples, desired_samples)


    def testReadTextFiletemperature(self):
        got_samples =  len(self.eda2.Temperature())
        desired_samples = 90028
        self.assertEqual(got_samples, desired_samples)

   
    
    def testDownsampling(self):
	got_samples = len(self.eda3.downsample(8).data)
        desired_samples = 24219
	self.assertEqual(got_samples, desired_samples)
	


if __name__ == "__main__":
    unittest.main()
