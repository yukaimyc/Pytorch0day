import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size()) ##[]表示的是维度数据
x,y = Variable(x),Variable(y)
#神经网络只能输入Variable类型的数据

#下面这两行代码可以看到神经网络生成的图长什么样子
plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()



class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output): #构造函数
        #构造函数里面的三个参数分别为，输入，中间隐藏层处理，以及输出层
        super(Net,self).__init__() #官方步骤
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predit=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):  #搭建的第一个前层反馈神经网络  向前传递
        x = F.relu(self.hidden(x))
        x = self.predit(x)  #此行可预测也可以不预测
        return x



net = Net(1,10,1)
#print(net)  #//此行用于观看到是否网络搭建成功，产生效果
###下面代码实现可视化
plt.ion()   #实现图像的实时打印
plt.show()

optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
loss_func = torch.nn.MSELoss() #使用均方差处理回归问题


for t in range(100):
    prediction =net(x)

    loss = loss_func(prediction,y) #//预测值一定要在前面，真实值要在后面

    optimizer.zero_grad() #将所有参数的梯度全部降为0，梯度值保留在这个里面
    loss.backward()    #反向传递过程
    optimizer.step()   #优化梯度

    if t%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f' % loss.data,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)