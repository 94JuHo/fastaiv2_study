from fastai.vision.all import *
from fastbook import *
from matplotlib import pyplot as plt

time = torch.arange(0, 20).float()
print(time)

speed = torch.randn(20) * 3 + 0.75*(time-9.5)**2 + 1
#plt.scatter(time, speed)
#plt.show()

def f(t, params):
    a, b, c = params
    return a*(t**2) + (b*t) + c

def mse(preds, targets):
    return ((preds - targets)**2).mean()

params = torch.randn(3).requires_grad_()
orig_params = params.clone()

preds = f(time, params)

def show_preds(preds, ax=None):
    if ax is None:
        ax = plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300, 100)

show_preds(preds)
#plt.show()

loss = mse(preds, speed)
print(loss)

loss.backward()
print(params.grad)
print(params.grad * 1e-5)
print(params)

lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None

preds = f(time, params)
print(mse(preds, speed))

show_preds(preds)
#plt.show()

def apply_step(params, prn=True):
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn:
        print(loss.item())
    return preds

for _ in range(10):
    apply_step(params)

params = orig_params.detach().requires_grad_()
_, axs = plt.subplots(1, 4, figsize=(12, 3))
for ax in axs:
    show_preds(apply_step(params, False), ax)
plt.tight_layout()
#plt.show()
