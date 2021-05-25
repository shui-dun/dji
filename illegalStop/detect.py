from torch.utils.data import DataLoader
from torchvision import datasets

from mixin import *

if __name__ == '__main__':
    path = r"D:\file\code\PROJECTS\djiDetect\dji\resource\cars"
    dataset = datasets.ImageFolder(path, transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    # 加载模型
    net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    net.load_state_dict(torch.load("models/last.pth"))

    with torch.no_grad(), open('output/result.txt', 'w') as f:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for val in predicted:
                f.write('{}\n'.format(val))
