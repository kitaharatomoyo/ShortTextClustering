import matplotlib.pyplot as plt
import numpy as np
import copy
import random

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 8, (3, 300))
        self.pool = nn.MaxPool2d(1, 2)
        self.fc1 = nn.Linear(8 * 9 * 1, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 8 * 9 * 1)
        y = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(y))
        return x, y

def batch_generator(mat, batch_size):
    mat = copy.copy(mat)
    n_batches = mat.shape[0] // batch_size
    mat = mat[:batch_size * n_batches,:]

    random.shuffle(mat)
    for n in range(0, mat.shape[0], batch_size):
        x = mat[n:n + batch_size,:6000]
        y = mat[n:n + batch_size,6000:]
        yield x, y

if __name__ == '__main__':

    dict = unpickle('/home/huang/Code/NLP/CNN_pytorch/data/data_batch_1')
    input, labels = dict[b'data'], dict[b'labels']
    input = input / 256

    data = np.concatenate((input, np.array([labels]).T), axis=1)

    net = Net()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        generator = batch_generator(data, 10)
        i = 0
        for inputs, labels in generator:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(torch.Tensor(inputs.reshape(-1, 3, 32, 32)))
            loss = criterion(outputs, torch.LongTensor(labels.reshape(-1)))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            i += 1
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0

    with torch.no_grad():
        dict = unpickle('/home/huang/Code/NLP/CNN_pytorch/data/test_batch')
        print(dict.keys())
        input, labels = dict[b'data'], dict[b'labels']
        input = input / 256

        outputs = net(torch.Tensor(input.reshape(-1, 3, 32, 32)))
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == torch.tensor(labels)).sum().item()
    
        total = len(labels)
    
    print('Acc = %d %%' % (100 * correct / total))


