from fastai.vision.all import *
from fastbook import *
from matplotlib import pyplot as plt

def f(x):
    return x**2

plot_function(f, 'x', 'x**2')
#plt.show()

plot_function(f, 'x', 'x**2')
plt.scatter(-1.5, f(-1.5), color='red')
#plt.show()

xt = tensor(3.).requires_grad_()

yt = f(xt)
print(yt)

yt.backward()
print(xt.grad)

xt = tensor([3., 4., 10.]).requires_grad_()
print(xt)

def f(x):
    return (x**2).sum()

yt = f(xt)
print(yt)

yt.backward()
print(xt.grad)