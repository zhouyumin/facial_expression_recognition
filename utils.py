# @Time : 2021/6/10 21:14 
# @Author : zym
# @File : utils.py 
# @Software: PyCharm
# @Description: 工具方法
import matplotlib.pyplot as plt
import pandas as pd
# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

text_labels = ['生气', '厌恶', '恐惧', '高兴', '难过', '惊讶', '中立']


def show_batch_img(sample_batch, title) -> None:
    images, labels, = sample_batch
    size = images.size(0)

    fig = plt.figure()
    for index, img in enumerate(images):
        ax = fig.add_subplot(size // 4, 4, index + 1)
        ax.axis('off')
        ax.imshow(img.numpy().transpose(1, 2, 0), cmap='gray')
        ax.set_title(text_labels[labels[index]])
    fig.suptitle(title)
    fig.show()


def show_loss() -> None:
    with open('./models/loss.txt', 'r', encoding='utf-8') as f:
        losses = f.readlines()
    losses = list(map(float, losses))

    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.plot(losses[0::200])
    plt.show()


def data_info(path):
    df = pd.read_csv(path, names=['img', 'label'])
    info = df['label'].value_counts(sort=False).tolist()
    plt.figure()
    plt.bar(text_labels, info)
    plt.show()


if __name__ == '__main__':
    show_loss()
