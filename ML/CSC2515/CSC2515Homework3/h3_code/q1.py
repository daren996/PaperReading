import random
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def get_data(w_true_, n=500, d=5, e=10):
    """
    Get data from a vector of given weights.

    :param n: number of samples of X
    :param d: number of features of X
    :param w_true_: the true w vector
    :param e: scale of error
    :return x_: a NxD matrix
    :return t_: a N-dimension vector
    """
    x_ = np.random.random((n, d)) * 2 - 1
    x_ = np.concatenate((x_, np.ones((n, 1))), axis=1)
    error_ = np.random.random((n,)) * e - e / 2
    t_ = np.dot(x_, w_true_) + error_
    t_ = t_.reshape(n, 1)
    print("x shape:", x_.shape, "t shape:", t_.shape)
    return x_, t_


w_truth = np.array([3, 5, -9, 28, -10, 1])
x, y = get_data(np.array(w_truth), n=500, d=w_truth.shape[0] - 1)
# x, y = [], []
# for i in range(1000):
#     temp = np.random.rand(2) * 10
#     x.append(temp)
#     y.append(np.sort(temp))
x_data = torch.from_numpy(np.array(x))
y_data = torch.from_numpy(np.array(y))
x_data = Variable(torch.tensor(x_data, dtype=torch.float32), requires_grad=False)
y_data = Variable(torch.tensor(y_data, dtype=torch.float32), requires_grad=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.w1 = Variable(torch.Tensor([[1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]), requires_grad=True)
        # self.w2 = Variable(torch.Tensor([[1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]]), requires_grad=True)
        self.w1 = Variable(torch.Tensor([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]]), requires_grad=True)
        self.w2 = Variable(torch.Tensor([[1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]]), requires_grad=True)
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        # in_size = x.size(0)
        # x = x.view(in_size, -1)  # flatten the tensor
        # x = self.fc2(self.sigmoid(self.fc1(x)))
        # x = self.fc2(self.fc1(x))
        x = x.mm(self.w1)
        x = self.relu(x)
        x = x.mm(self.w2)
        return x


model = Net()

print(model(Variable(torch.tensor(torch.from_numpy(np.array([[-1, 2]])), dtype=torch.float32))))
print(model(Variable(torch.tensor(torch.from_numpy(np.array([[5, 3]])), dtype=torch.float32))))
print(model(Variable(torch.tensor(torch.from_numpy(np.array([[5, 9]])), dtype=torch.float32))))


# optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08,
#                           weight_decay=0, momentum=0.001, centered=False)
# optimizer = optim.Adagrad(model.parameters(), lr=0.1, lr_decay=0, weight_decay=0,
#                           initial_accumulator_value=0)
optimizer = optim.SGD(model.parameters(), lr=0.0000001, momentum=0.01)
criterion = torch.nn.MSELoss(size_average=False)

for i in range(10000):
    output = model(x_data)
    loss = criterion(output, y_data)
    print(i, loss.data)  # , output, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


params = list(model.named_parameters())
print(params)


# print(model(Variable(torch.tensor(torch.from_numpy(np.array([[1, 2]])), dtype=torch.float32))))
# print(model(Variable(torch.tensor(torch.from_numpy(np.array([[5, 3]])), dtype=torch.float32))))
# print(model(Variable(torch.tensor(torch.from_numpy(np.array([[7, 1]])), dtype=torch.float32))))

aaa = [[1, 3, -21, 10, 5, 1]]
print(model(Variable(torch.tensor(torch.from_numpy(np.array(aaa)), dtype=torch.float32))))
print(np.dot(np.array(aaa), w_truth))
