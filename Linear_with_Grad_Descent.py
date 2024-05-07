import torch
"Computing a Linear Regression model using Gradient Descent Optimiser from scratch"

torch.manual_seed(3)

# Dataset
x = torch.tensor([2,3,4,5,6,7], dtype= torch.float32)
y = torch.tensor([4,6,8,10,12,14], dtype= torch.float32)
m = torch.randn(1, dtype= torch.float32) # Gradient
c = torch.randn(1, dtype= torch.float32) # Constant
h = torch.tensor([0.01], dtype= torch.float32) # Step size

# Linear Regression model
def f(gradient, input, constant):
    return torch.mul(gradient, input) #+ constant

# Computing the derivative of loss wrt to parameters
#def dldm(y_true, gradient, input, constant, step):
    change = (torch.mean(y_true - ((gradient + step)* input) + (constant))**2)
    mean = (torch.mean(y_true - (gradient * input) + constant) ** 2)
    return (change - mean) / step

def dldm(y_true, gradient, input, step):
    change = torch.mean((y_true - (gradient + step)* input)**2)
    mean = torch.mean((y_true - (gradient * input)) ** 2)
    return (change - mean) / step

#def dldc(y_true, gradient, input, constant, step):
    change = (torch.mean(y_true - ((gradient) * input) + (constant + step))**2)
    mean = (torch.mean(y_true - (gradient * input) + constant) ** 2)
    return (change - mean) / step

def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

def gradient_descent(parameter, deriavtive, lr):
    return parameter - (lr * deriavtive)

# Class definition for model
class Linear_Regression():
    def __init__(self):
        pass

    def forward(self, x):
        out = f(m,x,c)
        return out

iterations = 800
lr = 0.001
model = Linear_Regression()

for epoch in range(iterations):
    prediction = model.forward(x)
    loss = mse(y, prediction)

    grad_m = dldm(y, m, x, h)
    #grad_c = dldc(y, m , x , c , h)

    optimized_m = gradient_descent(m, grad_m, lr)
    m = optimized_m

    #optimized_c = gradient_descent(c, grad_c, lr)
    #c = optimized_c

    print(f"Epoch {epoch + 1}, Loss: {loss:.3f}, {m.item():.3f}, {grad_m.item():.3f}")




