# shuffle ,seperate data(mode), batch, recieve data
import numpy as np
import torch


## hyper parameter
Batch_size = 64


class Dataloader:
    def loader(self,dataset,mode,batch_size = None,shuffle = True):
        ratio = 0.7
        if shuffle:
            np.random.shuffle(dataset.numpy())
            dataset = torch.Tensor(dataset)

        if mode == 'Train':
            trainbatchdata = []
            traindata = dataset[0:int(ratio*len(dataset))]
            iterations = int(np.ceil(len(traindata)/batch_size))
            for i in range(iterations-1):
                trainbatchdata.append(traindata[i*batch_size:(i+1)*batch_size])
            trainbatchdata.append(traindata[(iterations-1)*batch_size:])
            return trainbatchdata
        elif mode == 'Test':
            testdata = dataset[int(ratio*len(dataset)):]
            return testdata




