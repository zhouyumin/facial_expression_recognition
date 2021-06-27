# @Time : 2021/6/10 21:04 
# @Author : zym
# @File : data_load.py
# @Software: PyCharm
# @Description: 加载数据集
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from DataSet import DataSet

weights = torch.FloatTensor([1.0266, 9.4066, 1.0010, 0.5684, 0.8491, 1.2934, 0.8260])

batch_size = 32

# 归一化
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机翻转
    transforms.RandomRotation(degrees=(-90, 90)),
    transforms.ToTensor(),
])
transform = transforms.Compose([
    transforms.ToTensor(),
])


# 训练集
train_dataset = DataSet(path='./data/train.csv', transform=train_transform)
train_targets = train_dataset.get_labels()
samples_weights = weights[train_targets]
train_sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)

# 校验集
val_dataset = DataSet(path='./data/val.csv', transform=transform)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# 测试集
test_dataset = DataSet(path='./data/test.csv', transform=transform)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    from utils import show_batch_img

    data = next(iter(train_dataloader))
    show_batch_img(data, '展示')
