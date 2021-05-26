import os
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

batch_size = 128

n_epoch = 200

lr = 0.001

datasetPath = 'data/lineData'

transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 交叉熵损失函数，常用于分类问题
weight = torch.zeros(1000)
weight[0] = 0.05
weight[1] = 0.95
criterion = nn.CrossEntropyLoss(weight=weight)


class MyDateset(Dataset):
    """
    自己实现数据集，继承于Dateset
    """

    def __init__(self, transforms=None, path=datasetPath, isTrain=True):
        self.transforms = transforms
        if isTrain:
            fileName = '{}/train.txt'.format(path)
        else:
            fileName = '{}/test.txt'.format(path)
        self.lst = []
        with open(fileName) as f:
            for line in f:
                self.lst.append(line.split())

    # 数据集的大小
    def __len__(self):
        return len(self.lst)

    # 数据集中第item个元素,返回格式[图像Tensor, label]
    def __getitem__(self, item):
        # img = read_image(self.lst[item][0]).byte()
        img = Image.open(self.lst[item][0])
        if self.transforms is not None:
            img = self.transforms(img)
        return [img, int(self.lst[item][1])]


# 将数据集切分为train和test
def split(originPath="D:/file/code/PROJECTS/djiDetect/100MEDIA/cars/turn1/result"):
    lst = []
    cls = 0
    for subDir in os.listdir(originPath):
        for file in os.listdir('{}/{}'.format(originPath, subDir)):
            lst.append(['{}/{}/{}'.format(originPath, subDir, file), cls])
        cls += 1
    random.shuffle(lst)
    lst_train = lst[:int(len(lst) * 0.8)]
    lst_test = lst[int(len(lst) * 0.8):]
    with open('{}/train.txt'.format(datasetPath), 'w') as f:
        for line in lst_train:
            f.write('{} {}\n'.format(line[0], line[1]))
    with open('{}/test.txt'.format(datasetPath), 'w') as f:
        for line in lst_test:
            f.write('{} {}\n'.format(line[0], line[1]))


def count(outputs, labels):
    _, predicted = torch.max(outputs.data[:, :2], 1)
    total = labels.size(0)
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    tn = ((predicted == 0) & (labels == 0)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()
    return total, tp, tn, fp, fn


def estimate(total, tp, tn, fp, fn, total_loss):
    acc = (tp + tn) / total
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * r * p / (r + p)
    loss = total_loss / total
    return acc, p, r, f1, loss

if __name__ == '__main__':
    split()
