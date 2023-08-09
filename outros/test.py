import numpy as np

def runge_kutta_4th_order(f, x0, y0, x_max, n):
    h = (x_max - x0) / n
    x_values = np.linspace(x0, x_max, n+1)
    y_values = np.zeros(n+1)
    
    y_values[0] = y0
    
    for i in range(n):
        x = x_values[i]
        y = y_values[i]
        
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        
        y_values[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return x_values, y_values

def f(x, y):
    return x + y

x0 = 0
x_final = 2
y0 = 1
n = 5

x_values, y_values = runge_kutta_4th_order(f, x0, y0, x_final, n)

for i in range(n+1):
    print(f"x = {x_values[i]:.2f}, y = {y_values[i]:.6f}")
