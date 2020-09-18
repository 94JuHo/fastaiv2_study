from fastai.vision.all import *
from fastbook import *
from matplotlib import pyplot as plt

matplotlib.rc('image', cmap='Greys')

path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path
print(path.ls())
print((path/'train').ls())

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
print(threes)

im3_path = threes[1]
im3 = Image.open(im3_path)
#im3.show()

print(array(im3)[4:10, 4:10])
print(tensor(im3)[4:10, 4:10])

im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15, 4:22])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
pd.set_option('display.width', None)
print(df)

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
print(len(three_tensors), len(seven_tensors))

show_image(three_tensors[1])
#plt.show()

stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
print(stacked_threes.shape)
print(len(stacked_threes.shape))
print(stacked_threes.ndim)

mean3 = stacked_threes.mean(0)
show_image(mean3)
#plt.show()

mean7 = stacked_sevens.mean(0)
show_image(mean7)
#plt.show()

a_3 = stacked_threes[1]
show_image(a_3)
#plt.show()

dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
print(dist_3_abs, dist_3_sqr)
print(F.l1_loss(a_3.float(), mean7), F.mse_loss(a_3, mean7).sqrt())

data = [[1, 2, 3], [4, 5, 6]]
arr = array(data)
tns = tensor(data)

print(arr)
print(tns)
print(tns[1])
print(tns[:, 1])
print(tns[1, 1:3])
print(tns+1)
print(tns.type())
print(tns*1.5)