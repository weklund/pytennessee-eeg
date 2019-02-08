import numpy as np
import pandas as pd
from pandas import DataFrame
import pyedflib
import re
import sys
import os
import math
import json
from DataSets.BaseDataset import BaseDataset


class TUHdataset(BaseDataset):
    """
    This class contains several methods that can process the EDF files in the TUH dataset.
    It serves 2 purposes:
    1) to create (and later load) a json file that contains the metadata for the entire TUH data.
    2) to read a single EDF file and extract signal values from that file.
    """

    def __init__(self, rootDir, xlsxFilePath):
        """
        Constructor
        """
        self.recordInfo = record_info
        self.numEdfs = num_edfs
        self.patientInfo = patient_info
        self.numPatients = num_patients
        self.rootDir = rootDir
        print("Top level directory for the dataset = ", rootDir)
        self.xlsxFilePath = xlsxFilePath
        print("xlsx File = ", xlsxFilePath)

    def summarizeDatset(self):
        """
        Print various summary information about the dataset, based on the root directory.

        Filename:
            edf/dev_test/01_tcp_ar/002/00000258/s002_2003_07_21/00000258_s002_t000.edf

            Components:
            edf: contains the edf data

            dev_test: part of the dev_test set (vs.) train

            01_tcp_ar: data that follows the averaged reference (AR) configuration,
                        while annotations use the TCP channel configuration

            002: a three-digit identifier meant to keep the number of subdirectories
                in a directory manageable. This follows the TUH EEG v1.1.0 convention.

            00000258: official patient number that is linked to v1.1.0 of TUH EEG

            s002_2003_07_21: session two (s002) for this patient. The session
                            was archived on 07/21/2003.

            00000258_s002_t000.edf: the actual EEG file. These are split into a series of
                        files starting with t000.edf, t001.edf, ... These
                        represent pruned EEGs, so the original EEG is
                        split into these segments, and uninteresting
                        parts of the original recording were deleted
                        (common in clinical practice).

            The easiest way to access the annotations is through the spreadsheet
            provided (_SEIZURES_*.xlsx). This contains the start and stop time
            of each seizure event in an easy to understand format. Convert the
            file to .csv if you need a machine-readable version.
        """
        # First summarize from the directory
        num_patients = 0
        num_patient_sessions = 0
        num_edfs = 0
        patient_info = {}
        record_info = {}
        # traverse root directory, and list directories as dirs and files as files
        for root, dirs, files in os.walk(self.rootDir):
            path_components = root.split(os.sep)
            # print ("root = ", root)
            if re.search("^\d\d\d$", path_components[-1]):
                # print ("List of patients = ", dirs)
                num_patients += len(dirs)
                for patient_id in dirs:
                    patient_info[patient_id] = {}
                    patient_info[patient_id]['sessions'] = []
                    patient_info[patient_id]['records'] = []
            if path_components[-1] in patient_info.keys():
                patient_id = path_components[-1]
                for session_id in dirs:
                    patient_info[patient_id]['sessions'].append(session_id)
                    num_patient_sessions += 1
            for filename in files:
                if re.search("\.edf$", filename) is not None:
                    edf_file_path = os.path.join(root, filename)
                    patient_id = path_components[-2]
                    session_id = path_components[-1]
                    record_id = os.path.basename(edf_file_path)
                    record_id = os.path.splitext(record_id)[0]
                    patient_info[patient_id]['records'].append(record_id)
                    record_info[record_id] = {}
                    record_info[record_id]['edf_file_path'] = edf_file_path
                    record_info[record_id]['patient_id'] = patient_id
                    record_info[record_id]['session_id'] = session_id
                    num_edfs += 1
            # print((len(path_components) - 1) * '---', os.path.basename(root))
            # for file in files:
            #     print(len(path_components) * '---', file)
        print("Total number of patients = ", num_patients)
        print("number of unique patients = ", len(patient_info))
        print("number of patient sessions = ", num_patient_sessions)
        print("Total number of EDFs = ", num_edfs)

        for record_id in self.recordInfo.keys():
            self.getEdfSummary(record_id)

        # for patient_id in patient_info.keys():
        #     for record_id in patient_info[patient_id]['records']:
        #         print (self.record_info[record_id])

    def getEdfSummary(self, recordID):
        filePath = self.recordInfo[recordID]['edfFilePath']
        f = pyedflib.EdfReader(filePath)
        numChannels = f.signals_in_file
        channelLabels = f.getSignalLabels()
        otherLabels = []
        columnsToDel = []
        for i in np.arange(numChannels):
            if (re.search('\-REF', channelLabels[i]) == None):
                otherLabels.append(channelLabels[i])
                columnsToDel.append(i)
                numChannels -= 1
        self.recordInfo[recordID]['channelLabels'] = np.delete(channelLabels, columnsToDel, axis=0).tolist()
        self.recordInfo[recordID]['other_labels'] = otherLabels
        self.recordInfo[recordID]['numChannels'] = np.int32(numChannels).item()
        self.recordInfo[recordID]['numSamples'] = np.int32(f.getNSamples()[0]).item()
        self.recordInfo[recordID]['sampleFrequency'] = np.int32(f.getSampleFrequency(0)).item()

    def getRecordData(self, recordID):
        filePath = self.recordInfo[recordID]['edfFilePath']
        numChannels = self.recordInfo[recordID]['numChannels']
        numSamples = self.recordInfo[recordID]['numSamples']
        channelLabels = self.recordInfo[recordID]['channelLabels']
        sigbufs = np.zeros((numChannels, numSamples))
        f = pyedflib.EdfReader(filePath)
        for i in np.arange(numChannels):
            try:
                sigbufs[i, :] = f.readSignal(i)
            except ValueError:
                print("Failed to read channel {} with name {}".format(i, channelLabels[i]))
        # sigbufs above is a 23 x 921600 matrix
        # transpose it so that it becomes 921600 x 23 matrix
        sigbufs = sigbufs.transpose()
        f._close()
        del (f)
        # print (sigbufs)
        return (sigbufs)

    def getSeizuresSummary(self):
        '''
        Read the xlsx file and summarize the seizure information on per-record basis
        '''
        with open(self.xlsxFilePath, 'rb') as f:
            df_out = pd.read_excel(f, sheet_name='train', usecols="A:O", dtype=object)

        print(df_out)
        filenames = df_out['Filename']
        filenameCol = df_out.columns.get_loc('Filename')
        seizureStartCol = filenameCol + 1
        seizureEndCol = seizureStartCol + 1
        seizureTypeCol = seizureEndCol + 1
        print("seizureStartCol = {}, seizureEndCol = {}".format(seizureStartCol, seizureEndCol))
        recordIDs = []
        rowIndex = -1
        for filename in filenames:
            rowIndex += 1
            if (not isinstance(filename, str)):
                continue
            # print ("filename = ", filename)
            if (re.search('\.tse$', filename) != None):
                recordID = os.path.basename(filename)
                recordID = os.path.splitext(recordID)[0]
                # try:
                seizureStartTime = df_out.iloc[rowIndex, seizureStartCol]
                seizureEndTime = df_out.iloc[rowIndex, seizureEndCol]
                seizureType = df_out.iloc[rowIndex, seizureTypeCol]
                if (not math.isnan(seizureStartTime)):
                    self.recordInfo[recordID]['seizureStart'] = np.float32(seizureStartTime).item()
                    self.recordInfo[recordID]['seizureEnd'] = np.float32(seizureEndTime).item()
                    self.recordInfo[recordID]['seizureType'] = seizureType
                # except:
                #     print ("rowIndex = ", rowIndex)
                # recordIDs.append(recordID)
        # print (recordIDs)
        # Get the column index for 'Seizure Time'
        # for patientID in self.patientInfo.keys():
        #     for recordID in self.patientInfo[patientID]['records']:
        #         print (self.recordInfo[recordID])

    def saveToJsonFile(self, filePath):
        print("Saving to the json file ", filePath)

        with open(filePath, 'w') as f:
            f.write("{\n")
            for patientID in self.patientInfo.keys():
                for recordID in self.patientInfo[patientID]['records']:
                    try:
                        f.write("\"" + recordID + "\" : ")
                        f.write(json.dumps(self.recordInfo[recordID]))
                        f.write(",\n")
                    except TypeError:
                        print("Record = ", self.recordInfo[recordID])
            f.write("\"EOFmarker\" : \"EOF\" }\n")

    def loadJsonFile(self, filePath):
        print("Loading from json file ", filePath)
        f = open(filePath, 'r')
        self.recordInfo = json.load(f)
        f.close()
        del (self.recordInfo['EOFmarker'])

        # Build self.patientInfo
        self.numEdfs = len(self.recordInfo)
        patientInfo = {}
        for recordID in self.recordInfo.keys():
            patientID = self.recordInfo[recordID]['patientID']
            sessionID = self.recordInfo[recordID]['sessionID']
            if (patientID not in patientInfo.keys()):
                patientInfo[patientID] = {}
                patientInfo[patientID]['records'] = [recordID]
                patientInfo[patientID]['sessions'] = [sessionID]
            else:
                patientInfo[patientID]['records'].append(recordID)
                patientInfo[patientID]['sessions'].append(sessionID)
        #     print ("self.recordInfo[" + recordID + "][\'numChannels\'] = ", self.recordInfo[recordID]['numChannels'])
        self.patientInfo = patientInfo
        self.numPatients = len(self.patientInfo)

    def isSeizurePresent(self, recordID, epochNum, epochLen, slidingWindowLen):
        '''
        epochLen and slidingWindowLen are in milliseconds
        '''
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return False
        # Convert epochNum to start and end datetime objects
        #         print ("recordID = ", recordID, ", epochNum =", epochNum, ", epochLen = ", epochLen,
        #                ", slidingWindowLen = ", slidingWindowLen)
        epochStart = float(epochNum * slidingWindowLen / 1000)
        epochEnd = epochStart + float(epochLen / 1000)

        seizureStart = self.recordInfo[recordID]['seizureStart']
        seizureEnd = self.recordInfo[recordID]['seizureEnd']
        # seizureType = self.recordInfo[recordID]['seizureType']

        if (((epochStart >= seizureStart) and (epochStart <= seizureEnd)) or
                ((epochEnd >= seizureStart) and (epochEnd <= seizureEnd))):
            return True

        return False

    def recordContainsSeizure(self, recordID):
        '''
        Returns True if there is at least one seizure entry in the entire record
                False otherwise
        '''
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return False
        else:
            return True

    def getExtendedSeizuresVectorCSV(self, recordID, epochLen, slidingWindowLen, numEpochs, priorSeconds, postSeconds):
        '''
        epochLen and slidingWindowLen are in milliseconds
        '''
        print("priorSeconds={}, postSeconds={}, epochLen={}, slidingWindowLen={}, numEpochs={}".format(priorSeconds,
                                                                                                       postSeconds,
                                                                                                       epochLen,
                                                                                                       slidingWindowLen,
                                                                                                       numEpochs))
        seizuresVector = np.zeros((numEpochs), dtype=np.int32)
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return seizuresVector

        seizureStart = self.recordInfo[recordID]['seizureStart']
        seizureEnd = self.recordInfo[recordID]['seizureEnd']
        # Update the seizureStart and seizureEnd values with the prio and post seconds
        seizureStart = max(0, seizureStart - priorSeconds)
        # seizureEnd may have a value t hat is more than the EDF duration,
        #  but that does not hurt in this method
        seizureEnd += postSeconds
        print("recordID={}, seizureStart={},seizureEnd={}".format(recordID, seizureStart, seizureEnd))
        # seizureType = self.recordInfo[recordID]['seizureType']
        for i in range(numEpochs):
            epochStart = float(i * slidingWindowLen / 1000)
            epochEnd = epochStart + float(epochLen / 1000)
            if (((epochStart >= seizureStart) and (epochStart <= seizureEnd)) or
                    ((epochEnd >= seizureStart) and (epochEnd <= seizureEnd))):
                seizuresVector[i] = 1
        # print (seizuresVector)
        return (seizuresVector)

    def getSeizuresVectorCSV(self, recordID, epochLen, slidingWindowLen, numEpochs):
        '''
        epochLen and slidingWindowLen are in milliseconds
        '''

        seizuresVector = np.zeros((numEpochs), dtype=np.int32)
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return seizuresVector
        else:
            seizureStart = self.recordInfo[recordID]['seizureStart']
            seizureEnd = self.recordInfo[recordID]['seizureEnd']
            # seizureType = self.recordInfo[recordID]['seizureType']
            for i in range(numEpochs):
                epochStart = float(i * slidingWindowLen / 1000)
                epochEnd = epochStart + float(epochLen / 1000)
                if (((epochStart >= seizureStart) and (epochStart <= seizureEnd)) or
                        ((epochEnd >= seizureStart) and (epochEnd <= seizureEnd))):
                    seizuresVector[i] = 1
            return seizuresVector

    def getSeizuresVectorEDF(self, recordID):
        numSamples = self.recordInfo[recordID]['numSamples']
        sampleFrequency = self.recordInfo[recordID]['sampleFrequency']
        seizuresVector = np.zeros((numSamples), dtype=np.int32)
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return seizuresVector
        else:
            seizureStart = self.recordInfo[recordID]['seizureStart']
            seizureEnd = self.recordInfo[recordID]['seizureEnd']
            # seizureType = self.recordInfo[recordID]['seizureType']
            for i in range(numSamples):
                sampleStartTime = float(i / sampleFrequency)
                sampleEndTime = float((i + 1) / sampleFrequency)
                if (((sampleStartTime >= seizureStart) and (sampleStartTime <= seizureEnd)) or
                        ((sampleEndTime >= seizureStart) and (sampleEndTime <= seizureEnd))):
                    seizuresVector[i] = 1
            return seizuresVector

    def getSeizureStartEndTimes(self, recordID):
        seizureStart = self.recordInfo[recordID]['seizureStart']
        seizureEnd = self.recordInfo[recordID]['seizureEnd']
        return (seizureStart, seizureEnd)

    def countRowsForEDFDataSubset(self, recordID, priorSeconds, postSeconds):
        if ('seizureStart' in self.recordInfo[recordID].keys()):
            (seizureStart, seizureEnd) = self.getSeizureStartEndTimes(recordID)
        else:
            return (0)  # This record has no seizure data

        numRows = self.recordInfo[recordID]['numSamples']
        numFeatures = self.recordInfo[recordID]['numChannels']
        print("numRows = ", numRows, ", numFeatures = ", numFeatures)
        startSec = seizureStart - priorSeconds
        endSec = seizureEnd + postSeconds
        startRowNum = int(startSec * self.recordInfo[recordID]['sampleFrequency'])
        endRowNum = int(endSec * self.recordInfo[recordID]['sampleFrequency'])
        if (startRowNum < 0):
            startRowNum = 0
        if (endRowNum > numRows):
            endRowNum = numRows
        numRows = endRowNum - startRowNum + 1
        print("numRows = ", numRows)
        return (numRows)

    # def countRowsForCSVDataSubset(self, recordID, epochLen, slidingWindowLen, numEpochs, priorSeconds, postSeconds):
    #     '''
    #     epochLen and slidingWindowLen are in milliseconds
    #     '''
    #     if ('seizureStart' not in self.recordInfo[recordID].keys()):
    #         return (0) # This record has no seizure data

    #     seizureStart = self.recordInfo[recordID]['seizureStart']
    #     seizureEnd = self.recordInfo[recordID]['seizureEnd']
    #     # Update the seizureStart and seizureEnd values with the prio and post seconds
    #     seizureStart = max(0, seizureStart-priorSeconds)
    #     # seizureEnd may have a value t hat is more than the EDF duration,
    #     #  but that does not hurt in this method
    #     seizureEnd += postSeconds
    #     # seizureType = self.recordInfo[recordID]['seizureType']
    #     numRows = 0
    #     for i in range(numEpochs):
    #         epochStart = float(i * slidingWindowLen / 1000)
    #         epochEnd = epochStart + float(epochLen / 1000)
    #         if (( (epochStart >= seizureStart) and (epochStart <= seizureEnd)) or
    #             ( (epochEnd >= seizureStart) and (epochEnd <= seizureEnd))):
    #                 numRows += 1
    #     return (numRows)

    def getEDFDataSubset(self, recordID, priorSeconds, postSeconds):
        dataset = self.getRecordData(recordID)
        print(dataset)
        if ('seizureStart' in self.recordInfo[recordID].keys()):
            (seizureStart, seizureEnd) = self.getSeizureStartEndTimes(recordID)
        else:
            return (None)  # This record has no seizure data

        numRows = dataset.shape[0]
        numFeatures = self.recordInfo[recordID]['numChannels']
        print("numRows = ", numRows, ", numFeatures = ", numFeatures)
        startSec = seizureStart - priorSeconds
        endSec = seizureEnd + postSeconds
        startRowNum = int(startSec * self.recordInfo[recordID]['sampleFrequency'])
        endRowNum = int(endSec * self.recordInfo[recordID]['sampleFrequency'])
        if (startRowNum < 0):
            startRowNum = 0
        if (endRowNum > numRows):
            endRowNum = numRows
        numRows = endRowNum - startRowNum + 1
        print("numRows = ", numRows)
        dataset = dataset[startRowNum:endRowNum + 1]
        return (dataset)

    def getCSVDataSubset(self, recordID, csv_df, seizuresVector):
        '''
        csv_df -- data frame corresponding to the CSV file
        '''
        if ('seizureStart' not in self.recordInfo[recordID].keys()):
            return (None)  # This record has no seizure data

        startRowNum = 0
        endRowNum = csv_df.shape[0] - 1
        for i in range(len(seizuresVector)):
            if (seizuresVector[i] == 1):
                if (startRowNum <= 0):
                    startRowNum = i
                if (endRowNum != i):
                    endRowNum = i
        dataset = csv_df.iloc[startRowNum:endRowNum + 1]
        print("Original dataset shape = {}, reduced dataset shape = {}".format(csv_df.shape, dataset.shape))
        return (dataset)
