##########################################################
# pytorch-qnn v1.0                                     
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import torch
from torch                import nn
from torch.autograd       import Variable
from torch.utils.data     import DataLoader
from torchvision          import transforms
from torchvision.utils    import save_image
from torchvision.datasets import CIFAR10
from convolutional_models import CAE, QCAE
import os
from scipy                import misc
import numpy              as np
import sys


def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)

if len(sys.argv) > 1:
    model       = str(sys.argv[1])
else:
    print("Please provide a model : QCAE or CAE")
    exit(0)

cuda            = True
num_epochs      = 3000
learning_rate   = 0.0005
generation_rate = 100 # One test picture will be generated every 'generation_rate'

if model == 'QCAE':
    net  = QCAE()
else:
    net  = CAE()

if cuda:
    net = net.cuda()

#
# MANAGING PICTURES
#

os.system('rm -rf RES/test_image*')
train = rgb2gray(misc.imread('KODAK/kodim05.png'))
test  = misc.imread('KODAK/kodim23.png')

# Normalizing
train = train / 255
test  = test  / 255

#
# HYPERPARAMETERS
#

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

nb_param  = sum(p.numel() for p in net.parameters() if p.requires_grad)

print("QCAE & CAE Color images - Titouan Parcollet - LIA, ORKIS")
print("Model Info --------------------")
print("Number of trainable parameters : "+str(nb_param))

if model == 'QCAE':

    # Add a 0 as real component if quaternions (padding)
    npad  = ((0, 0), (0, 0), (1, 0))
    train = np.pad(train, pad_width=npad, mode='constant', constant_values=0)
    # Channel first
    train = np.transpose(train, (2,0,1))
    # Add batch_size dim
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))

    
    test = np.pad(test, pad_width=npad, mode='constant', constant_values=0)
    test = np.transpose(test, (2,0,1))
    test = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))
    
else:
    # Channel first
    train = np.transpose(train, (2,0,1))
    # Add batch_size dim
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))
    
    test = np.transpose(test, (2,0,1))
    test = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))

train = torch.from_numpy(train).float().cuda()
test  = torch.from_numpy(test).float().cuda()

for epoch in range(num_epochs):
    
    output = net(train)
    loss   = criterion(output, train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("It : "+str(epoch+1)+" | loss_train "+str(loss.cpu().data[0].numpy()))
    
    # If generation rate, generate a test image
    if (epoch %generation_rate) == 0:
        output = net(test)
        out    = output.cpu().data.numpy()
        if model == 'QCAE':
            out = np.transpose(out, (0,2,3,1))[:,:,:,1:]
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))
        else:
            out = np.transpose(out, (0,2,3,1))
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))

        misc.imsave("RES/save_image"+str(epoch)+".png", out)


