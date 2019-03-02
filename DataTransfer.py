import numpy as np
import torch
import logging
import sys
import dataloader
import dataset
import transform
import os
import copy

Batch_size = 32
imageType = ['Gray', 'Colour']
dataAddress = './data/'
saveAddress2 = '%s/PreprocessData2.0'%dataAddress
dataType = ['frequency_domain', 'time_domain']
dataAddr = '%s/labeleddata_%s.pkl'%(saveAddress2, dataType[0])
type = 'Gray'
mode = 'Test'


D = dataloader.Dataloader()
p = dataset.Dataset(data_transform=transform.horizontalFlip())


# load processed data from pickle,if the data not exist, it will take longer to process first
if not os.path.isfile(dataAddr):
    dataAddr = p.preprocessing(dataType=dataType[0],DataperSample=1300)  # data with label
else:
    labeldata = torch.load(dataAddr)

trainData = D.loader(dataset=labeldata,batch_size=Batch_size,
                     mode='Train',shuffle=True,)  # 16*89*1301
testData = D.loader(dataset=labeldata,batch_size=Batch_size,
                     mode='Test',shuffle=True,)


class transfer:
    def __init__(self, dataset, mode):
        super(transfer, self).__init__()
        self.dataset = dataset
        self.mode = mode

    def toImage(self, height, type, normalize = True):
        coeff = 1 if normalize == False else 255
        data1 = self.dataset
        transferedData = []
        if self.mode == 'Train':
            for _, batchContent in enumerate(data1):
                #number of samples in each batch
                length = len(batchContent)
                if type == 'Gray': #begin transform to gray images
                    labels = batchContent[:,-1]
                    images = torch.Tensor([])
                    for r in range(length):    #for each sample, transform it to image
                        sampleData =batchContent[r][:-1]
                        totalPixel = len(sampleData)
                        width = totalPixel // height
                        buffer = sampleData[:height * width] #receive current data according to specific size
                        buffer = coeff *(buffer - min(buffer)) /(max(buffer) - min(buffer))  #mapping data within 255 or just normalize
                        buffer = buffer.reshape(height, -1)
                        images = torch.cat((images,buffer.unsqueeze(0)),0) #transform each sequence into a image
                    transferedData.append((images.unsqueeze(1),labels))

                elif type == 'Colour': #begin to transform to colour images
                    labels = batchContent[:,-1]
                    images = torch.Tensor([])
                    for r in range(length):  # number of samples in each batch
                        sampleData =batchContent[r][:-1]
                        totalPixel = len(sampleData)
                        step = totalPixel // 3 #ensure the size of each channel
                        r_channel = torch.Tensor([])
                        g_channel = torch.Tensor([])
                        b_channel = torch.Tensor([])
                        for n in range(step):
                            r_channel = torch.cat((r_channel,sampleData[3 * n].unsqueeze(0)),0)
                            try:
                                g_channel = torch.cat((g_channel, sampleData[3 * n+1].unsqueeze(0)), 0)
                            except:
                                pass
                            try:
                                b_channel = torch.cat((b_channel, sampleData[3 * n+2].unsqueeze(0)), 0)
                            except:
                                pass
                        totalPixel = len(r_channel)
                        newheight = int(height/1.732)
                        width = totalPixel // newheight
                        r_channel = r_channel[:width*newheight].reshape(newheight,-1)
                        g_channel = g_channel[:width*newheight].reshape(newheight,-1)
                        b_channel = b_channel[:width*newheight].reshape(newheight,-1)

                        image = torch.cat((r_channel.unsqueeze(0),g_channel.unsqueeze(0),b_channel.unsqueeze(0)),0)
                        imagecopy = copy.deepcopy(image)
                        images = torch.cat((imagecopy.unsqueeze(0),images),0)
                    transferedData.append((images,labels))
                else:
                    logging.error('please check the type name carefully and restart the program')
                    sys.exit()
        else:
            labels = data1[:,-1]
            images = torch.Tensor([])
            for sample in data1:
                if type == 'Gray': #begin transform to gray images
                    sampleData =sample[:-1]
                    totalPixel = len(sampleData)
                    width = totalPixel // height
                    buffer = sampleData[:height * width] #receive current data according to specific size
                    buffer = coeff * (buffer - min(buffer)) / (
                                max(buffer) - min(buffer))  # mapping data within 255 or just normalize
                    buffer = buffer.reshape(height, -1)
                    images = torch.cat((images,buffer.unsqueeze(0)),0) #transform each sequence into a image

                elif type == 'Colour': #begin to transform to colour images
                    images = torch.Tensor([])
                    sampleData =sample[:-1]
                    totalPixel = len(sampleData)
                    step = totalPixel // 3 #ensure the size of each channel
                    r_channel = torch.Tensor([])
                    g_channel = torch.Tensor([])
                    b_channel = torch.Tensor([])
                    for n in range(step):
                        r_channel = torch.cat((r_channel,sampleData[3 * n].unsqueeze(0)),0)
                        try:
                            g_channel = torch.cat((g_channel, sampleData[3 * n+1].unsqueeze(0)), 0)
                        except:
                            pass
                        try:
                            b_channel = torch.cat((b_channel, sampleData[3 * n+2].unsqueeze(0)), 0)
                        except:
                            pass
                    image = torch.cat((r_channel.unsqueeze(0),g_channel.unsqueeze(0),b_channel.unsqueeze(0)),0)
                    imagecopy = copy.deepcopy(image)
                    images = torch.cat((imagecopy.unsqueeze(0),images),0)
                else:
                    logging.error('please check the type name carefully and restart the program')
                    sys.exit()
            transferedData = (images,labels)

        return transferedData


t = transfer(testData)  #train data needs to be matched with train mode
transferData = t.toImage(50,type = type)
print(transferData)
# torch.save(transferData,'%s/transfer_%s_%s.pkl'%(saveAddress2,type,mode,))

# import visdom
# vis = visdom.Visdom()
#
# transferData,label = transferData[0]
# #single image show
# vis.image(
#     transferData[0],
#     opts=dict(title='myImage', caption='p1'),
# )
