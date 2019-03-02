import dataloader
import dataset
from torch import nn
import numpy as np
import torch
import os
import os.path
# import visdom
import transform


# hyper parameter
datapersample = 1300
LR = 0.001
EPOCH = 200
Batch_size = 128
hidden_size = 30

dataAddress = './data/'
saveAddress2 = '%s/PreprocessData2.0'%dataAddress
dataType = ['frequency_domain', 'time_domain']
dataAddr = '%s/labeleddata_%s.pkl'%(saveAddress2, dataType[0])

D = dataloader.Dataloader()
p = dataset.Dataset(data_transform=transform.horizontalFlip())


# load processed data from pickle,if the data not exist, it will take longer to process first
if not os.path.isfile(dataAddr):
    dataAddr = p.preprocessing(dataType=dataType[0],DataperSample=datapersample)  # data with label

labeldata = torch.load(dataAddr)
traindata = D.loader(dataset=labeldata,batch_size=Batch_size,
                     mode='Train',shuffle=True,)  # 16*89*1301
testdata = D.loader(dataset = labeldata,mode = 'Test',shuffle= False)

test_x = testdata[:,:-1].view(-1,50,26)
test_y = testdata[:,-1]

class RNN(nn.Module):
    def __init__(self,Hidden_size,Biflag):
        super(RNN, self).__init__()

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

        self.out = nn.Linear(self.hidden_dim*2, 9)

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


rnn = RNN(hidden_size,Biflag=False)


# training and testing
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=0.0005 )   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
losses = []
axis = 0
# vis = visdom.Visdom()

for epoch in range(EPOCH):
    for step, data in enumerate(traindata):        # gives batch data
        b_x = data[:,:-1]
        b_y = data[:,-1]
        b_x = b_x.view(-1, 50, 26)        # reshape x to (batch, time_step, input_size)
        b_x = torch.autograd.Variable(b_x)

        output = rnn(b_x,batch_size=Batch_size)                               # rnn output
        loss = loss_func(output, b_y.long())                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        # # save loss
        # losses.append(loss.data.numpy())

        if step % 5 == 0:
            test_output = rnn(test_x,batch_size=len(test_x))                   # (samples, time_step, input_size)
            _, pred_y = torch.max(test_output, 1)
            accuracy = (pred_y == test_y.long()).sum().numpy() / test_y.size(0)
            testloss = loss_func(test_output, test_y.long())
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            # vis.line(X=np.array([axis]),
            #          Y=np.array([loss.data.numpy()]),
            #          win='window',
            #          update='append',
            #          # opts = dict(linecolor = np.array([[0,0,0]]) )
            #          )
            # vis.line(X=np.array([axis]),
            #          Y=np.array([testloss.data.numpy()]),
            #          win='window',
            #          update='append',
            #          # opts = dict(linecolor = np.array([[0,0,0]]) )
            #          )
            # axis += 1

# if you choose to save loss result
saveadd = '%s/figure/lossresult/'%dataAddress
if not os.path.exists(saveadd):
    os.makedirs(saveadd)
torch.save(losses, '%s/hiddensize%s.pkl'%(saveadd,hidden_size))


# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 50,26 ))
pred_y = torch.max(test_output, 1)[1]
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
