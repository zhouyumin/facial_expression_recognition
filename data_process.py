# @Time : 2021/6/8 8:12 
# @Author : zym
# @File : data_process.py 
# @Software: PyCharm
# @Description: 数据集预处理
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
import pandas as pd
import numpy as np
import cv2


def depart() -> list:
    """
    分离feature和label
    :return:
    """
    df = pd.read_csv('./data/fer2013/fer2013.csv')
    feature = df['pixels']
    label = df['emotion']
    usage = df['Usage']
    label.to_csv('./data/label.csv', index=False, header=False)
    feature.to_csv('./data/feature.csv', index=False, header=False)

    return usage.value_counts().values.tolist()


def to_img() -> None:
    """
    将csv像素值转成灰度图
    :return:
    """
    data = np.loadtxt('./data/feature.csv')  # 读取像素
    for i in range(data.shape[0]):
        face = data[i, :]
        face = face.reshape(48, 48)  # reshape
        cv2.imwrite('./data/face_img/{}.jpg'.format(i), face)  # 写图片


def split_data(train, public_test, private_test) -> None:
    """
    划分训练集、测试集和校验集
    :return:
    """
    df = pd.read_csv('./data/label.csv', header=None)
    train_img = ['{}.jpg'.format(i) for i in range(train)]
    test_img = ['{}.jpg'.format(i) for i in range(train, train + public_test)]
    val_img = ['{}.jpg'.format(i) for i in range(train + public_test, df.shape[0])]

    train_label = df.iloc[:train, 0]
    test_label = df.iloc[train:train + public_test, 0]
    val_label = df.iloc[train + public_test:, 0]

    # 保存到训练集
    train = pd.DataFrame()
    train['path'] = pd.Series(train_img)
    train['label'] = train_label.values
    train.to_csv('./data/train.csv', index=False, header=False)

    # 保存到测试集
    test = pd.DataFrame()
    test['path'] = pd.Series(test_img)
    test['label'] = test_label.values
    test.to_csv('./data/test.csv', index=False, header=False)

    # 保存到校验集
    val = pd.DataFrame()
    val['path'] = pd.Series(val_img)
    val['label'] = val_label.values
    val.to_csv('./data/val.csv', index=False, header=False)


if __name__ == '__main__':
    nums = depart()
    to_img()
    split_data(*nums)
