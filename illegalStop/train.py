import plotly.graph_objects as go
import torch.optim as optim
import torch.utils.data

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
    net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
    # net.load_state_dict(torch.load("models/last.pth"))
    net.to(device)

    # 优化函数
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    x, y_loss, y_acc, y_p, y_r, y_f1 = [], [], [], [], [], []

    for epoch in range(n_epoch):
        total_loss, total_tp, total_tn, total_fp, total_fn, total = 0, 0, 0, 0, 0, 0
        for i, data in enumerate(trainloader):
            # 得到一个batch的输入和对应的标签
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)

            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            loss = loss.item()

            #  optimize
            optimizer.step()

            tt, tp, tn, fp, fn = count(outputs, labels)
            total += tt
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_loss += loss

        acc, p, r, f1, loss = estimate(total, total_tp, total_tn, total_fp, total_fn, total_loss)

        x.append(epoch)
        y_loss.append(loss)
        y_acc.append(acc)
        y_p.append(p)
        y_r.append(r)
        y_f1.append(f1)

        print('测试集上 loss: {:.3f} accuracy: {:.3f} p: {:.3f} r: {:.3f} f1: {:.3f}'.format(
            loss, acc, p, r, f1))
        # 保存模型
        PATH = './models/last_{}.pth'.format(epoch)
        torch.save(net.state_dict(), PATH)

        # 绘图
        fig = go.Figure(layout={
            "xaxis_title": "epoch"
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
        fig.write_html("output/{}.html".format(epoch))
