import numpy as np

def runge_kutta_4th_order(f, x0, y0, z0, x_max, n):
    h = (x_max - x0) / n
    x_values = np.linspace(x0, x_max, n+1)
    y_values = np.zeros(n+1)
    z_values = np.zeros(n+1)
    
    y_values[0] = y0
    z_values[0] = z0
    
    for i in range(n):
        x = x_values[i]
        y = y_values[i]
        z = z_values[i]
        
        k1y = h * z
        k1z = h * (f(x, y, z))
        
        k2y = h * (z + k1z/2)
        k2z = h * (f(x + h/2, y + k1y/2, z + k1z/2))
        
        k3y = h * (z + k2z/2)
        k3z = h * (f(x + h/2, y + k2y/2, z + k2z/2))
        
        k4y = h * (z + k3z)
        k4z = h * (f(x + h, y + k3y, z + k3z))
        
        y_values[i+1] = y + (k1y + 2*k2y + 2*k3y + k4y) / 6
        z_values[i+1] = z + (k1z + 2*k2z + 2*k3z + k4z) / 6
    
    return x_values, y_values, z_values

def f(x, y, z):
    return -2*y + x

x0 = 0
x_final = 10
y0 = 1
z0 = 0
n = 100

x_values, y_values, z_values = runge_kutta_4th_order(f, x0, y0, z0, x_final, n)

for i in range(n+1):
    print(f"x = {x_values[i]:.2f}, y = {y_values[i]:.6f}, z = {z_values[i]:.6f}")
