import unittest
from edatoolkit import qLogFile
from time import strftime
from datetime import datetime


class Toolkit_Test(unittest.TestCase):

    def setUp(self):
        self.ascii_file = "./resources/test.eda"

    def test_exportToCSV(self):
        header_line = 1
        eof_line = 1

        # load ascii eda file
        eda = qLogFile(self.ascii_file)

        # count the number of samples using the qLogFile property
        in_samples = eda.num_samples
        
        # export data to CSV (a List, really) and split on line-feeds
        out = eda.exportToCSV().split('\r\n')

        # count the samples less the header and eof lines
        out_samples = len(out) - header_line - eof_line

        self.assertEqual(out_samples, in_samples)
    
    def test_toString(self):
        header_lines = 8
        eof_line = 1
        markers = 0
        
        # load ascii eda file
        eda = qLogFile(self.ascii_file)
        
        # count the number of markers
        markers = len(eda.markers)

        # count the number of samples
        in_samples = eda.num_samples
        
        # export data to String and split on line-feeds
        out = eda.toString().split('\r\n')

        # count the samples less the header, eof, and markers
        out_samples = len(out) - header_lines - eof_line - markers
        
        self.assertEqual(out_samples, in_samples)

    def test_save(self):
        # declare and init needed vars
        header_lines = 8
        markers = 0
        out = None

        # load ascii eda file
        eda = qLogFile(self.ascii_file)
        
        # count the number of markers
        markers = len(eda.markers)

        # count the number of samples
        in_samples = eda.num_samples

        # save the data to a local file
        eda.save("./resources/test_save.eda")
        
        # open the saved file and read all the lines
        with open('./resources/test_save.eda', 'r') as f:
            out = f.readlines()
        
        # count the samples less the header and markers
        out_samples = len(out) - header_lines - markers

        self.assertEqual(out_samples, in_samples)

    def test_offsetForSeconds_01(self):
        test_offset = 1
        eda = qLogFile(self.ascii_file)
        offset = eda.offsetForSeconds(test_offset)
        self.assertTrue(offset)

    def test_offsetForSeconds_02(self):
        test_offset = 0
        eda = qLogFile(self.ascii_file)
        offset = eda.offsetForSeconds(test_offset)
        self.assertEqual(offset, test_offset)

    def test_offsetForSeconds_03(self):
        test_offset = -1
        eda = qLogFile(self.ascii_file)
        self.assertRaises(IndexError, eda.offsetForSeconds, test_offset)

    def test_offsetForSeconds_04(self):
        test_offset = 9999999999
        eda = qLogFile(self.ascii_file)
        self.assertRaises(IndexError, eda.offsetForSeconds, test_offset)

    def test_secondsForOffset_01(self):
        test_offset = 1
        eda = qLogFile(self.ascii_file)
        offset = eda.secondsForOffset(test_offset)
        self.assertIsNotNone(offset)

    def test_secondsForOffset_02(self):
        test_offset = 0
        eda = qLogFile(self.ascii_file)
        offset = eda.secondsForOffset(test_offset)
        self.assertEqual(offset, test_offset)

    def test_updateStartTime_01(self):
        test_time = "1974-12-13 13:13:13"
        eda = qLogFile(self.ascii_file)

        self.assertRaises(TypeError, eda.updateStartTime, test_time)
        
    def test_updateStartTime_02(self):
        test_time = datetime.today()
        eda = qLogFile(self.ascii_file)
        eda.updateStartTime(test_time)
        self.assertEqual(test_time, eda.startTime)

    def test_trim_01(self):
        test_start = 0
        test_end   = 0

        eda = qLogFile(self.ascii_file)

        eda = eda.trim(test_start, test_end)
        cut_samples = eda.num_samples

        self.assertEqual(cut_samples, 0)

    def test_trim_02(self):
        test_start = 0
        test_end   = 10

        eda = qLogFile(self.ascii_file)

        eda = eda.trim(test_start, test_end)
        cut_samples = eda.num_samples

        self.assertEqual(cut_samples, 10) 

    def test_trim_03(self):
        test_start = 0

        eda = qLogFile(self.ascii_file)
        total_samples = eda.num_samples

        eda = eda.trim(test_start, total_samples)
        cut_samples = eda.num_samples

        self.assertEqual(cut_samples, total_samples)

    def test_trim_04(self):
        test_start = 10
        test_end   = 0
        
        eda = qLogFile(self.ascii_file)
        
        eda = eda.trim(test_start, test_end)
        cut_samples = eda.num_samples
        
        self.assertEqual(cut_samples, 0)

    def test_downsample_01(self):
        test_rate = 0

        eda = qLogFile(self.ascii_file)

        self.assertRaises(ValueError, eda.downsample, test_rate)

    def test_downsample_02(self):
        test_rate = 32

        eda = qLogFile(self.ascii_file)

        self.assertRaises(ValueError, eda.downsample, test_rate)

    def test_downsample_03(self):
        test_rate = 0

        eda = qLogFile(self.ascii_file)

        test_rate = eda.sampleRate

        self.assertRaises(ValueError,eda.downsample, test_rate)

    def test_downsample_05(self):
        test_rate = 8

        eda = qLogFile(self.ascii_file)
        num_samples = eda.num_samples

        t_samples = num_samples / (32 / test_rate)

        downsampled_eda = eda.downsample(test_rate)
        new_samples = downsampled_eda.num_samples

        self.assertEqual(new_samples, t_samples)

    def test_downsample_06(self):
        test_rate = 35

        eda = qLogFile(self.ascii_file)
        
        self.assertRaises(ValueError, eda.downsample, test_rate)

    def offsetForTime_01(self):
        test_time = 0

        eda = qLogFile(self.ascii_file)

        self.assertRaises(TypeError, eda.offsetForTime, test_time)

    def offsetForTime_02(self):
        test_time = datetime.today()

        eda = qLogFile(self.ascii_file)
        self.assertRaises(TypeError, eda.offsetForTime, test_time)

    def test_edaSlice_01(self):
        test_start = 0
        test_end   = 0

        eda = qLogFile(self.ascii_file)

        self.assertRaises(IndexError, eda.edaSlice, test_start, test_end)

    def test_edaSlice_02(self):
        test_start = 0
        test_end   = 0

        eda = qLogFile(self.ascii_file)
        self.assertRaises(IndexError, eda.edaSlice, test_start, test_end)

    def test_edaSlice_03(self):
        test_start = 0
        test_end   = 1

        eda = qLogFile(self.ascii_file)

        test_slice = eda.edaSlice(test_start, test_end)

        eda_len = len(test_slice)

        self.assertEqual(eda_len, test_end)

    def test_edaSlice_04(self):
        test_start = datetime.today()
        test_end   = test_start

        eda = qLogFile(self.ascii_file)

        self.assertRaises(TypeError, eda.edaSlice, test_start, test_end)

    def test_loadLogFile_01(self):
        eda = qLogFile(self.ascii_file)

        self.assertRaises(TypeError, eda.loadLogFile, self.ascii_file)


if __name__ == '__main__':
    unittest.main()
