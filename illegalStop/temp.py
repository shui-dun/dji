import torch


if __name__ == '__main__':
    t = torch.Tensor([[7, 4, 7, 1],
        [1, 9, 0, 5],
        [8, 8, 8, 4]])
    print(t[:,:1] > 5)
