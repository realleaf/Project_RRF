import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import dataloader
import dataset
import transform
import os
import visdom
vis = visdom.Visdom()

# hyper parameter
gLearningRate = 0.001
dLearningRate = 0.001
BatchSize = 64
numEpoch = 10
DataperSample = 1300
hidden_size = 30

# data adress
dataAddress = './data/'
saveAddress2 = '%s/PreprocessData2.0'%dataAddress
dataType = ['frequency_domain', 'time_domain']
dataAddr = '%s/labeleddata_%s.pkl'%(saveAddress2, dataType[0])

# load data
D = dataloader.Dataloader()
p = dataset.Dataset(data_transform=transform.horizontalFlip())

# load processed data from pickle,if the data not exist, it will take longer to process first
if not os.path.isfile(dataAddr):
    dataAddr = p.preprocessing(dataType=dataType[0],DataperSample=DataperSample)  # data with label
labeldata = torch.load(dataAddr)

traindata = D.loader(dataset=labeldata,batch_size=BatchSize,mode='Train',shuffle=True,)  # 16*89*1301
testdata = D.loader(dataset = labeldata,mode = 'Test',shuffle= False)

test_x = testdata[:,:-1].view(-1,50,26)
test_y = testdata[:,-1]


# the Generator
class Generator(nn.Module):   # adadpted from AutoEncoder
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50*26,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,50*26),
            # nn.Sigmoid()    # FORCE OUTPUT INTO (0,1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded  # decoded can be used as generated data

GNet = Generator()
# print(Generator)


# the Discriminator
class Discriminator(nn.Module):
    def __init__(self,Hidden_size,Biflag):
        super(Discriminator, self).__init__()

        self.Biflag = Biflag
        self.num_layers = 2 if self.Biflag else 1
        self.hidden_dim = Hidden_size

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=26,
            hidden_size=self.hidden_dim,  # rnn hidden unit
            num_layers=self.num_layers,  # number of rnn layer
            batch_first=True,
            bidirectional= True
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(self.hidden_dim*2, 2)
        # self.out = nn.Sigmoid()# force the judge result falls in (0,1)

    def forward(self, x,batch_size):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        if self.Biflag:
            self.h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)
            self.c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)
            r_out, (h_n, h_c) = self.rnn(x,(self.h0,self.c0))
        else:
            r_out, (h_n, h_c) = self.rnn(x,)  # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


DNet = Discriminator(hidden_size,Biflag=False)
# print(Discriminator)


# DLoss = nn.BCEWithLogitsLoss()
DLoss = nn.CrossEntropyLoss()                       # the target label is not one-hotted

gOptimizer = torch.optim.Adam(GNet.parameters(), lr=gLearningRate,
                              betas=(0.9, 0.999))
dOptimizer = torch.optim.Adam(DNet.parameters(), lr=dLearningRate,
                              betas=(0.9, 0.999))     ######## how to set this paramaters


axis = 0
for epoch in range(numEpoch):
    for batch, realData in enumerate(traindata):
        realData =  realData[:,:-1].view(-1, 50, 26)
        randomData= Variable(torch.randn(realData.size()))
        realData = Variable(realData)

        # remake label
        # batch_size = realData.size(0)
        realDataLabels = Variable(torch.ones(realData.size(0)))
        fakeDataLabels = Variable(torch.zeros(realData.size(0)))

        # real data loss
        RealDataPreds = DNet(realData,batch_size=realData.size(0))    # Real data passing through RNN
        dLossReal = DLoss(RealDataPreds, realDataLabels.long())
        dMeanReal = RealDataPreds.mean()

        # fake data loss
        randomData = randomData.view(-1,50*26)
        FDataFeature,fakeData = GNet(randomData)
        fakeData = fakeData.view(-1,50,26)
        FakeDataPreds = DNet(fakeData,batch_size=realData.size(0))
        dLossFake = DLoss(FakeDataPreds, fakeDataLabels.long())   #### there is another different loss available
        dMeanFake = FakeDataPreds.sigmoid().mean()
        dLoss = dLossFake + dLossReal

        DNet.zero_grad()
        dLoss.backward(retain_graph = True)
        dOptimizer.step()

        #train generative net
        # gLoss = DLoss(FakeDataPreds, realDataLabels.long())
        # GNet.zero_grad()
        # gLoss.backward()
        # gOptimizer.step()

        FAKEDataPreds= DNet(fakeData,batch_size=realData.size(0))
        gLoss = DLoss(FAKEDataPreds, realDataLabels.long())
        gMeanFake = FAKEDataPreds.sigmoid().mean() # for showing result only
        GNet.zero_grad()
        gLoss.backward()
        gOptimizer.step()


        if batch % 5 == 0:
            print('Epoch:', epoch,
                  '| Discriminative loss: %.4f | Generative loss: %.4f | R2R ratio: %.2f | DemeanFake: %.4f '
                  %(dLoss, gLoss, dMeanReal,dMeanFake))

            vis.line(X=np.array([axis]),
                     Y=np.column_stack((np.array([dLoss.detach().numpy()]),
                                        np.array([gLoss.detach().numpy()]))),
                     win='win1',
                     update='append',
                     opts=dict(showlegend=True,
                               legend=['discriminative net loss',
                                'generative net loss']),
                     )
        axis += 1




