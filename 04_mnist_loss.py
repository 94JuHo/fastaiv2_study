from fastai.vision.all import *
from fastbook import *
from matplotlib import pyplot as plt

path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]

stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255

valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float() / 255
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float() / 255

train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)
train_y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)
print(train_x.shape, train_y.shape)

dset = list(zip(train_x, train_y))
x, y = dset[0]
print(x.shape, y)

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28*28)
valid_y = tensor([1] * len(valid_3_tens) + [0] * len(valid_7_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))

def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

weights = init_params((28 * 28, 1))
bias = init_params(1)

print((train_x[0] * weights.T).sum() + bias)

def linear1(xb):
    return xb@weights + bias

preds = linear1(train_x)
print(preds)

corrects = (preds > 0.0).float() == train_y
print(corrects)
print(corrects.float().mean().item())

weights[0] *= 1.0001
preds = linear1(train_x)
print( ((preds > 0.0).float() == train_y).float().mean().item())

trgts = tensor([1, 0, 1])
prds = tensor([0.9, 0.4, 0.2])

def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()

print(torch.where(trgts==1, 1-prds, prds))
print(mnist_loss(prds, trgts))
print(mnist_loss(tensor([0.9, 0.4, 0.8]), trgts))

def sigmoid(x):
    return 1/(1+torch.exp(-x))

plot_function(torch.sigmoid, title='Sigmoid', min=-4, max=4)
#plt.show()

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
print(list(dl))

ds = L(enumerate(string.ascii_lowercase))
print(ds)

dl = DataLoader(ds, batch_size=6, shuffle=True)
print(list(dl))

