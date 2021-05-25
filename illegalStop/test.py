import time

from mixin import *


def test(dataLoader, net, name):
    """
    测试准确度和耗时
    """
    total_loss, total_tp, total_tn, total_fp, total_fn, total = 0, 0, 0, 0, 0, 0

    t1 = time.time()
    num = 0
    with torch.no_grad():
        for data in dataLoader:
            images, labels = data
            num += len(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_tp += ((predicted == 1) == (labels == 1)).sum().item()
            total_tn += ((predicted == 0) == (labels == 0)).sum().item()
            total_fp += ((predicted == 1) == (labels == 0)).sum().item()
            total_fn += ((predicted == 0) == (labels == 1)).sum().item()
            total += labels.size(0)
    t2 = time.time()

    ls = total_loss / num
    acc = (total_tp + total_tn) / total
    p = total_tp / (total_tp + total_fp)
    r = total_tp / (total_tp + total_fn)
    f1 = 2 * r * p / (r + p)

    print('{}集上 loss: {:.3f} accuracy: {:.3f} p: {:.3f} r: {:.3f} f1: {:.3f}'.format(
        name, ls, acc, p, r, f1))


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
    net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    net.load_state_dict(torch.load("models/last.pth"))

    # 测试测试集
    test(testloader, net, "测试")
