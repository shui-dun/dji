import plotly.graph_objects as go
import torch.optim as optim
import torch.utils.data
from torch import nn

from mixin import *

if __name__ == '__main__':
    # 加载数据集
    trainset = MyDateset(
        transforms=transform,
        isTrain=True
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # 加载网络
    net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    net.load_state_dict(torch.load("models/last.pth"))
    net.to(device)

    # 交叉熵损失函数，常用于分类问题
    weight = torch.zeros(1000)
    weight[0] = 0.05
    weight[1] = 0.95
    criterion = nn.CrossEntropyLoss(weight=weight)
    # 更新参数
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(n_epoch):
        x, y_loss, y_acc, y_p, y_r, y_f1 = [], [], [], [], [], []
        total_loss, total_tp, total_tn, total_fp, total_fn, total = 0, 0, 0, 0, 0, 0
        for i, data in enumerate(trainloader):
            # 得到一个batch的输入和对应的标签
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)

            # 计算准确度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_tp += ((predicted == 1) == (labels == 1)).sum().item()
            total_tn += ((predicted == 0) == (labels == 0)).sum().item()
            total_fp += ((predicted == 1) == (labels == 0)).sum().item()
            total_fn += ((predicted == 0) == (labels == 1)).sum().item()

            # backward
            loss = criterion(outputs, labels)
            loss.backward()

            #  optimize
            optimizer.step()

            loss = loss.item()
            total_loss += loss

            num = 1
            if i % num == 0:
                # 打印结果
                ls = total_loss / num
                acc = (total_tp + total_tn) / total
                p = total_tp / (total_tp + total_fp)
                r = total_tp / (total_tp + total_fn)
                f1 = 2 * r * p / (r + p)
                x.append(i)
                y_loss.append(ls)
                y_acc.append(acc)
                y_p.append(p)
                y_r.append(r)
                y_f1.append(f1)
                print('[epoch:{}, batch{}] loss: {:.3f} accuracy: {:.3f} p: {:.3f} r: {:.3f} f1: {:.3f}'.format(
                    epoch, i, ls, acc, p, r, f1))
                total_loss, total_tp, total_tn, total_fp, total_fn, total = 0, 0, 0, 0, 0, 0

                # 保存模型
                PATH = './models/last.pth'
                torch.save(net.state_dict(), PATH)

        # 绘图
        fig = go.Figure(layout={
            "xaxis_title": "iteration"
        })
        fig.add_trace(go.Scatter(
            name="loss",
            x=x,
            y=y_loss,
            mode='lines',
        ))
        fig.add_trace(go.Scatter(
            name="accuracy",
            x=x,
            y=y_acc,
            mode='lines',
        ))
        fig.add_trace(go.Scatter(
            name="p",
            x=x,
            y=y_p,
            mode='lines',
        ))
        fig.add_trace(go.Scatter(
            name="r",
            x=x,
            y=y_r,
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            name="f1",
            x=x,
            y=y_f1,
            mode='lines'
        ))
        fig.show()
