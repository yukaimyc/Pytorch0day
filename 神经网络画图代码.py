
'''
np_data=np.arange(6).reshape((2,3))
#生成numpy数据
torch_data=torch.from_numpy(np_data) #numpy数据转化为torch数据
tensor2array=torch_data.numpy() #torch转化为numpy数据
print(
    '\nnumpy',np_data,
    '\ntorch',torch_data,
    '\ntensor2array',tensor2array,
)
#abs
data=[-1,-2,1,2]
tensor=torch.FloatTensor(data)
print(tensor)
'''
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

x=torch.linspace(-5,5,200)#x data
x=Variable(x)
x_np=x.data.numpy()

y_relu=F.relu(x).data.numpy()
y_sigmoid=F.sigmoid(x).data.numpy()
y_tanh=F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

##开始画图
plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')



plt.subplot(222)
plt.plot(x_np,y_relu,c='red',label='sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')


plt.subplot(223)
plt.plot(x_np,y_relu,c='red',label='tanh')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np,y_relu,c='red',label='softplus')
plt.ylim((-0.2,6))
plt.legend(loc='best')
plt.show()

