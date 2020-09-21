from fastai.vision.all import *
from fastbook import *
from matplotlib import pyplot as plt

def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

def linear1(xb):
    return xb@weights + bias

def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()

path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]

stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255

weights = init_params((28*28, 1))
bias = init_params(1)

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)

valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float() / 255
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float() / 255

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1] * len(valid_3_tens) + [0] * len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))

dset = list(zip(train_x, train_y))

dl = DataLoader(dset, batch_size=256)
xb, yb = first(dl)
print(xb.shape, yb.shape)

valid_dl = DataLoader(valid_dset, batch_size=256)

batch = train_x[:4]
print(batch.shape)

preds = linear1(batch)
print(preds)

loss = mnist_loss(preds, train_y[:4])
print(loss)

loss.backward()
print(weights.grad.shape, weights.grad.mean(), bias.grad)

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

calc_grad(batch, train_y[:4], linear1)
print(weights.grad.mean(), bias.grad)

calc_grad(batch, train_y[:4], linear1)
print(weights.grad.mean(), bias.grad)

weights.grad.zero_()
bias.grad.zero_()

def train_epoch(model, lr, params):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()

print((preds>0.0).float() == train_y[:4])

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()

print(batch_accuracy(linear1(batch), train_y[:4]))

def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

print(validate_epoch(linear1))

lr = 1.
params = weights, bias
train_epoch(linear1, lr, params)
print(validate_epoch(linear1))

for _ in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')

