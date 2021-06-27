# @Time : 2021/6/10 21:49 
# @Author : zym
# @File : train.py 
# @Software: PyCharm
# @Description: 训练
import torch
import torch.nn as nn
from torch import optim
from data_load import train_dataloader
import time
from ResNet18 import ResNet18
from test import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 20
lr = 0.001


net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(params=net.parameters(), lr=lr)

losses = []


def train() -> None:
    print('train on', device)
    better = 60
    for epoch in range(epochs):
        net.train()
        start = time.time()
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (i + 1) % 100 == 0:
                print('epoch: %d/%d, iter: %d/%d,  loss: %.4f'
                      % (epoch + 1, epochs, i + 1, len(train_dataloader), loss.item()))

        acc = test(net)
        if acc > better:
            better = acc
            torch.save(net.state_dict(), './models/better.pt')

        print('epoch: %d, loss: %.4f, train acc: %.2f%%, time: %.2f sec' % (epoch + 1, losses[-1], acc, time.time() - start))

    with open('./models/loss.txt', 'w', encoding='utf-8') as f:
        for loss in losses:
            f.write(str(loss) + '\n')

    print('training finish')


if __name__ == '__main__':
    train()
    torch.save(net.state_dict(), './models/model.pt')
