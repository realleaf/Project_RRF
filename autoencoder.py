import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

torch.manual_seed(1)

#hyper params

EPOCH = 10
BATCH_SIZE = 64
LR =0.005
DOWNLOAD = False
N_TEST_IMG  =5


#data
trainData = torchvision.datasets.MNIST\
            (root = './mnist/',train = True,
             transform = torchvision.transforms.ToTensor(),
             download = DOWNLOAD)
trainLoader = torch.utils.data.DataLoader(dataset = trainData,
                                          batch_size=BATCH_SIZE,shuffle = True)

# plot an example
# print(trainData.train_data.size())
# print(trainData.train_labels.size())
# plt.imshow(trainData.train_data[2].numpy(),cmap = 'gray')
# plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
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
            nn.Linear(128,28*28),
            nn.Sigmoid()    # FORCE OUTPUT INTO (0,1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded  # decoded can be used as generated data


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(),lr = LR)
Loss = nn.MSELoss()

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = trainData.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())


for epoch in range(EPOCH):
    for step,(x,y) in enumerate(trainLoader):
        b_x = Variable(x.view(-1,28*28))
        b_y = Variable(x.view(-1,28*28))
        b_label = Variable(y)

        encoded, decoded = autoencoder(b_x)

        loss = Loss(decoded,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step %100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); plt.pause(0.05)


plt.ioff()
plt.show()

# visualize in 3D plot
view_data = trainData.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2)
ax = Axes3D(fig)

X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = trainData.train_labels[:200].numpy()

for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, s, backgroundcolor=c)

ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()



