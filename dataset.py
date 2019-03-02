import torch
import numpy as np
import pandas as pd
from itertools import chain
from functools import reduce
import operator
import os

dataAddress = './data/'
saveAddress = '%s/PreprocessData1.0'%dataAddress
saveAddress2 = '%s/preprocessdata'%dataAddress
dataType = ['frequency_domain', 'time_domain']
mode = ['train', 'test']
numFile = 10
numClass = 8
tunnal_IQ = ['I', 'Q']
datapersample = 1300  # number of data for sample

# primary preprocess method
class Dataset:
    def __init__(self,data_transform = None):
        self.transform = data_transform

    def saveAs(self, dataAddress, dataType, numFile, numClass, saveAddress,
               tunnal_IQ = None):
        '''

        :param dataAddress: main data address
        :param dataType: different feature of data, time or frequency
        :param numFile: number of files in each class
        :param numClass: number of class in each feature dimension
        :param saveAddress: the address to save data
        :param tunnal_IQ: specify which tunnel you want to extract
        :result: the concatenated data for all class and all data under one class
        '''
        #determine the filename in path
        fileName = 'spectrum' if dataType == 'frequency_domain' else 'IQ'

        #determine the tunnal when process time domain data
        IQ = tunnal_IQ if dataType == 'time_domain' else None

        #determine the column of data to select
        column = [3] if IQ == 'Q' else [1]             ### could be specified by user instead

        ### DISCUSS: how about the order of two for-recurrent ? ? ? C and F###
        for C in range(numClass):
            for F in range(numFile):
                buffer = pd.read_excel('%s/%s/exp%d/exp%d_%s%d.xlsx'%
                                       (dataAddress, dataType, C + 1, C + 1, fileName, F + 1),
                                       usecols=column)

                #joint multiplt data files into 'dataReceiver' by column
                if not 'dataReceiver' in dir():
                    dataReceiver = buffer
                else:
                    dataReceiver = np.r_[dataReceiver, buffer]
                print('file%s in class%s has finished'%(F, C))
        #ensure the path is right
        if not os.path.exists(saveAddress):
            os.makedirs(saveAddress)
        torch.save(dataReceiver, '%s/%s_class%s.pkl'%(saveAddress, dataType, numClass))


    def preprocessing(self,dataType,DataperSample):
        '''

        :param dataType: what type of data you are dealing with, could be time dormain data or frequency dormain
        :param DataperSample: how many data in a sample you want
        :param data_transform: use transform function written as a class to processing data
        :return: allLabeldata: a data matrix where the last column is the label (tensor)

        '''
        data = torch.load('%s/%s_class%s.pkl'%(saveAddress, dataType, numClass))
        datapoint = len(data)//numClass
        samplenum = datapoint//DataperSample

        for c in range(numClass):
            classdata = data[c*datapoint:c*datapoint+DataperSample*samplenum]
            # Normalize data within each class
            classdata = (classdata - min(classdata)) / (max(classdata) - min(classdata))
            classdata = (classdata-np.mean(classdata))/np.std(classdata,ddof=1)

            classdata = classdata.reshape(-1,DataperSample)
            datawithlabel = np.c_[classdata,[c]*len(classdata)]
            if c == 0:
                allLabelData = datawithlabel
            else:
                allLabelData = np.r_[allLabelData, datawithlabel]

        if self.transform is not None:
            allLabelData = self.transform(torch.Tensor(allLabelData))

        if not os.path.exists(saveAddress2):
            os.makedirs(saveAddress2)
        torch.save(allLabelData,'%s/labeledData_%s.pkl'%(saveAddress2, dataType))

        return '%s/labeleddata_%s.pkl'%(saveAddress2, dataType)


# p = Dataset(data_transform=transform.Transforms.horizontalFlip())
# p.saveAs(dataAddress,dataType = 'frequency_domain', numFile=numFile, numClass=numClass, saveAddress=saveAddress,tunnal_IQ = 'I')
# labeldata = p.preprocessing(dataType='time_domain',DataperSample=datapersample)
# print(labeldata.size())
# print(labeldata)
