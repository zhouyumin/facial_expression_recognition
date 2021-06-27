# @Time : 2021/6/10 20:45 
# @Author : zym
# @File : DataSet.py
# @Software: PyCharm
# @Description: 重写Dataset
from torch.utils import data
import pandas as pd
import numpy as np
from PIL import Image


class DataSet(data.Dataset):
    # 初始化
    def __init__(self, path, transform):
        super(DataSet, self).__init__()
        self.transform = transform
        df_path = pd.read_csv(path, header=None, usecols=[0])
        df_label = pd.read_csv(path, header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    # 读取某幅图片，index为索引号
    def __getitem__(self, index):
        # 读取灰度图
        img = Image.open('./data/face_img/' + self.path[index])

        # 变换成Tensor
        img = self.transform(img)

        label = self.label[index]

        return img, label

    # 获取数据集样本个数
    def __len__(self):
        return self.path.shape[0]

    def get_labels(self):
        return self.label
