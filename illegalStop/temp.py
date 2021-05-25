import torchvision
import torch

if __name__ == '__main__':
    weight = torch.zeros(1000)
    weight[0] = 0.05
    weight[1] = 0.95
    print(weight)

