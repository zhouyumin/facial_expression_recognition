# @Time : 2021/6/10 21:54 
# @Author : zym
# @File : test.py 
# @Software: PyCharm
# @Description: 测试
import torch
from data_load import test_dataloader, train_dataloader, val_dataloader
from utils import show_batch_img
from ResNet18 import ResNet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(net, dataloader=val_dataloader) -> float:
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).int().sum().item()
        return 100 * correct / total


def test_img_batch(net) -> None:
    images, labels = next(iter(test_dataloader))
    images = images[:16]  # 目前只测试16张
    labels = labels[:16]
    inputs = images.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    show_batch_img((images, labels), '真实结果')
    show_batch_img((images, predicted), '预测结果')


def main():
    net = ResNet18().to(device)
    net.load_state_dict(torch.load('./models/better.pt'))
    # test_img_batch(net)
    print('accuracy of the model of the val images: %.2f%%' % test(net, test_dataloader))


if __name__ == '__main__':
    main()
    # net = ResNet18().to(device)
    # net.load_state_dict(torch.load('./models/model.pt'))
    # test_img_batch(net)
