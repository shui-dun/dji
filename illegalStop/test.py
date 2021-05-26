import time

from mixin import *


def test(dataLoader, net):
    """
    测试准确度和耗时
    """
    total_loss, total_tp, total_tn, total_fp, total_fn, total = 0, 0, 0, 0, 0, 0

    t1 = time.time()
    with torch.no_grad():
        for data in dataLoader:
            images, labels = data
            outputs = net(images)
            tt, tp, tn, fp, fn = count(outputs, labels)
            total += tt
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_loss += criterion(outputs, labels)
    t2 = time.time()

    acc, p, r, f1, loss = estimate(total, total_tp, total_tn, total_fp, total_fn, total_loss)

    print('测试集上 loss: {:.3f} accuracy: {:.3f} p: {:.3f} r: {:.3f} f1: {:.3f} 耗时: {:.3f}s'.format(
        loss, acc, p, r, f1, t2 - t1))


if __name__ == '__main__':
    # 数据集
    testset = MyDateset(
        transforms=transform,
        isTrain=False
    )

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 加载模型
    net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
    net.load_state_dict(torch.load("models/last.pth"))

    # 测试测试集
    test(testloader, net)
