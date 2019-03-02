import torch

class horizontalFlip(object):

    def __call__(self, data):

        flipdata = torch.flip(data[:,:-1],[1])
        flipdata = torch.cat((flipdata,data[:,-1].unsqueeze(1)),1)
        data = torch.cat((data,flipdata),0)

        return data

    def randomCrop(self,data,dim):
        pass



