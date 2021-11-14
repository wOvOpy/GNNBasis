import torch
'''
1. 准备数据集
'''
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
'''
2. 使用用类设计模型
'''
class LinearModel(torch.nn.Module): 
    def __init__(self):
        super(LinearModel, self).__init__() 
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x): 
        y_pred = self.linear(x) 
        return y_pred

'''
3. 构建损失函数和优化器
'''
model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
'''
4. 训练周期
'''
for epoch in range(1000):
    # 前向传播
    y_pred = model.forward(x_data)
    # 计算损失函数
    loss = criterion(y_pred, y_data) 
    print(epoch, loss.item())
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
print('w = ', model.linear.weight.item()) 
print('b = ', model.linear.bias.item())
x_test = torch.Tensor([[4.0]]) 
y_test = model(x_test) 
print('y_pred = ', y_test.data)