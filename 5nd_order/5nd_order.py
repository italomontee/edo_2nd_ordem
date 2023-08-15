import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# 1 ORDEM ############

def plot_graph_first_order():
    x_values, y_values = solve_edo1(equation_entry1.get(), g)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x")
    plt.legend()
    plt.grid(True)
    plt.show()

def iniciar_1nd_ordem_i(gb):
    p = int(p_entry1.get())
    
    x = Symbol('x')
    y = Function('y')(x)

    equation_str = equation_entry1.get()

    eq1 = eval(equation_str)

    x_v, y_v = solve_edo1(equation_str, gb)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label1.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def iniciar_1nd_ordem_l(gb):
    p = int(p_entry1.get())

    x = Symbol('x')
    y = Function('y')(x)

    equation_str = equation_entry1.get()

    eq1 = eval(equation_str)

    x_v, y_v = solve_edo1(equation_str, gb)

    text = ""
    
    for i in range(0, p+1):
        text += f"{[i]} y1: {y_v[i]:.6f}\n"
    
    result_text1.delete('1.0', tk.END)
    result_text1.insert(tk.END, text)
    
def solve_edo1(equation_str, gb):
    global g
    g = gb

    # Obter os valores informados pelo usuário
    x0 = float(x0_entry1.get())
    x_final = float(x0_final_entry1.get())
    y0 = float(y0_entry1.get())
    n = int(np_entry1.get())
    
    derivative_str = equation_str

    # Definir a variável simbólica para x e y
    x, y = symbols('x y')

    # Definir a função f(x, y) como a derivada de y (y')
    derivative_expr = sympify(derivative_str)
    f = lambdify((x, y), derivative_expr, modules=['numpy'])

    # Resolvendo a equação usando o método de Runge-Kutta
    if gb == 4:
        x_values, y_values = runge_kutta_4th_order_edo_1th_order(f, x0, y0, x_final, n)
    elif gb == 6:
        x_values, y_values = runge_kutta_6th_order_edo_1th_order(f, x0, y0, x_final, n)
    elif gb == 1:
        x_values, y_values = solve_euler1(f, x0, y0, x_final, n)
    elif gb == 2:
        x_values, y_values = solve_heun1(f, x0, y0, x_final, n)

    return x_values, y_values

def solve_heun1(f, x0, y0, x_max, n):
    step = (x_max-x0)/n
    
    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    while x < x_max:
        k1 = step * f(x, y)
        k2 = step * f(x + step, y + k1)
        y = y + 0.5 * (k1 + k2)
        x = x + step
        x_values.append(x)
        y_values.append(y)

    return x_values, y_values

def solve_euler1(f, x0, y0, x_max, n):
    step = (x_max-x0)/n
    
    x_values = [x0]
    y_values = [y0]


    x = x0
    y = y0

    while x < x_max:
        y = y + step * f(x, y)
        x = x + step
        x_values.append(x)
        y_values.append(y)

    return x_values, y_values

def runge_kutta_4th_order_edo_1th_order(f, x0, y0, x_max, n):
    # Implementar o método de Runge-Kutta de quarta ordem aqui
    # Retornar os valores de x e y para o intervalo de interesse
    h = (x_max-x0)/n
    x_values = np.linspace(x0, x_max, n+1)
    y_values = np.zeros(n+1)

    # Definir os valores iniciais
    x_values[0] = x0
    y_values[0] = y0

    # Método de Runge-Kutta de quarta ordem
    for i in range(1, n+1):
        x = x_values[i - 1]
        y = y_values[i - 1]
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)

        y_values[i] = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x_values, y_values

def runge_kutta_6th_order_edo_1th_order(f, x0, y0, x_max, n):
    h = (x_max - x0) / n
    x_values = np.linspace(x0, x_max, n+1)
    y_values = np.zeros(n+1)

    x_values[0] = x0
    y_values[0] = y0

    for i in range(1, n+1):
        x = x_values[i - 1]
        y = y_values[i - 1]
        
        k1 = h * f(x, y)
        k2 = h * f(x + h/3, y + k1/3)
        k3 = h * f(x + 2*h/3, y + 2*k2/3)
        k4 = h * f(x + h, y + (k1 + 3*k2 + 3*k3) / 8)
        k5 = h * f(x + h/2, y + (k1 - 3*k2 + 4*k3 + 8*k4) / 16)
        k6 = h * f(x + h, y + (k1 + 4*k2 + k3 - 8*k4 + 2*k5) / 16)
        
        y_values[i] = y + (k1 + 4*k3 + k5 + 6*k6) / 15

    return x_values, y_values

# 2 ORDEM ############

def plot_graph_second_order():
    x_values, y_values , _= solve_edo2(equation_entry2.get(), g)

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x (2ª Ordem)")
    plt.legend()
    plt.grid(True)
    plt.show()

def iniciar_2nd_ordem_i(gb):
   
    # Solicitar informações do usuário
    p = int(p_entry2.get())

    #Definir simbolos
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)

    equation_str = equation_entry2.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v = solve_edo2(equation_str, gb)

    result_label2.config(text=f"y1 = {y_v[p]:.6f}, z1 = {z_v[p]:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def iniciar_2nd_ordem_l(gb):  
    p = int(p_entry2.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)

    equation_str = equation_entry2.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v = solve_edo2(equation_str, gb)
    

    text = ""
    
    for i in range(0, p+1):
        text += f"[{i}] \ny1: {y_v[i]:.6f} \nz1: {z_v[i]:.6f}\n\n"
    
    result_text2.delete('1.0', tk.END)
    result_text2.insert(tk.END, text)                  

def solve_edo2(equation_str, gb):
    global g
    g = gb

    # Obter os valores informados pelo usuário
    x0 = float(x0_entry2.get())
    x_final = float(x0_final_entry2.get())
    y0 = float(y0_entry2.get())
    z0 = float(z0_entry2.get())
    n = int(p_entry2.get())
    
    derivative_str = equation_str

    # Definir a variável simbólica para x e y
    x, y , dydx= symbols('x y dydx')

    # Definir a função f(x, y) como a derivada de y (y')
    derivative_expr = sympify(derivative_str)
    f = lambdify((x, y, dydx), derivative_expr, modules=['numpy'])

    # Resolvendo a equação usando o método de Runge-Kutta
    if gb == 4:
        x_values, y_values, z_values = runge_kutta_4th_order_edo_2th_order(f, x0, y0, z0, x_final, n)
    elif gb == 6:
        x_values, y_values, z_values = runge_kutta_6th_order_edo_2th_order(f, x0, y0, z0, x_final, n)
    elif gb == 1:
        x_values, y_values, z_values = solve_euler2(f, x0, y0, z0, x_final, n)
    elif gb == 2:
        x_values, y_values, z_values = solve_heun2(f, x0, y0, z0, x_final, n)

    return x_values, y_values, z_values

def solve_heun2(f, x0, y0, z0, x_max, n):
    step = (x_max-x0)/n
    
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]

    x = x0
    y = y0
    z = z0

    for _ in range(n):
        k1 = step * z
        l1 = step * f(x, y, z)

        k2 = step * (z + 0.5 * l1)
        l2 = step * f(x + 0.5 * step, y + 0.5 * k1, z + 0.5 * l1)

        y_next = y + k2
        yp_next = z + l2

        x = x + step
        y = y_next
        z = yp_next

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

    return x_values, y_values, z_values

def solve_euler2(f, x0, y0, z0, x_max, n):
    step = (x_max-x0)/n
    
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]

    x = x0
    y = y0
    yp = z0

    for _ in range(n):
        y_next = y + step * yp
        yp_next = yp + step * f(x, y, yp)

        x = x + step
        y = y_next
        z = yp_next

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

    return x_values, y_values, z_values

def runge_kutta_4th_order_edo_2th_order(f, x0, y0, y_prime0, x_max, n):

    h = (x_max - x0) / n
    x_values = np.linspace(x0, x_max, n + 1)
    y1_values = np.zeros(n + 1)
    y2_values = np.zeros(n + 1)

    x = x0
    y1 = y0
    y2 = y_prime0  

    for i in range(1, n + 1):
        k1_y1 = h * y2
        k1_y2 = h * f(x, y1, y2)

        k2_y1 = h * (y2 + 0.5 * k1_y2)
        k2_y2 = h * f(x + 0.5 * h, y1 + 0.5 * k1_y1, y2 + 0.5 * k1_y2)

        k3_y1 = h * (y2 + 0.5 * k2_y2)
        k3_y2 = h * f(x + 0.5 * h, y1 + 0.5 * k2_y1, y2 + 0.5 * k2_y2)

        k4_y1 = h * (y2 + k3_y2)
        k4_y2 = h * f(x + h, y1 + k3_y1, y2 + k3_y2)

        y1 = y1 + (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1) / 6
        y2 = y2 + (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 6

        x = x + h
        x_values[i] = x
        y1_values[i] = y1
        y2_values[i] = y2

    return x_values, y1_values, y2_values

def runge_kutta_6th_order_edo_2th_order(f, x0, y0, y_prime0, x_max, n):
    h = (x_max - x0) / n
    x_values = np.linspace(x0, x_max, n + 1)
    y1_values = np.zeros(n + 1)
    y2_values = np.zeros(n + 1)

    x = x0
    y1 = y0
    y2 = y_prime0

    for i in range(1, n + 1):
        k1_y1 = h * y2
        k1_y2 = h * f(x, y1, y2)

        k2_y1 = h * (y2 + k1_y2 / 3)
        k2_y2 = h * f(x + h / 3, y1 + k1_y1 / 3, y2 + k1_y2 / 3)

        k3_y1 = h * (y2 + 2 * k2_y2 / 3)
        k3_y2 = h * f(x + 2 * h / 3, y1 + 2 * k2_y1 / 3, y2 + 2 * k2_y2 / 3)

        k4_y1 = h * (y2 + k1_y2)
        k4_y2 = h * f(x + h, y1 + k1_y1, y2 + k1_y2)

        k5_y1 = h * (y2 + (k1_y2 + 4 * k2_y2 + k4_y2) / 6)
        k5_y2 = h * f(x + h / 2, y1 + (k1_y1 + 4 * k2_y1 + k4_y1) / 6, y2 + (k1_y2 + 4 * k2_y2 + k4_y2) / 6)

        k6_y1 = h * (y2 + (k1_y2 + 3 * k3_y2 + 4 * k4_y2 + k5_y2) / 8)
        k6_y2 = h * f(x + h, y1 + (k1_y1 + 3 * k3_y1 + 4 * k4_y1 + k5_y1) / 8, y2 + (k1_y2 + 3 * k3_y2 + 4 * k4_y2 + k5_y2) / 8)

        y1 = y1 + (k1_y1 + 3 * k3_y1 + 4 * k4_y1 + k5_y1 + 3 * k6_y1) / 15
        y2 = y2 + (k1_y2 + 3 * k3_y2 + 4 * k4_y2 + k5_y2 + 3 * k6_y2) / 15

        x = x + h
        x_values[i] = x
        y1_values[i] = y1
        y2_values[i] = y2

    return x_values, y1_values, y2_values

# 3 ORDEM ############

def plot_graph_third_order():
    x_values, y_values, _, _ = solve_edo3(equation_entry3.get(), gb)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x")
    plt.legend()
    plt.grid(True)
    plt.show()

def iniciar_3nd_ordem_i(gb):
    
    p = int(p_entry3.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)

    equation_str = equation_entry3.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v= solve_edo3(equation_str, gb)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label3.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def iniciar_3nd_ordem_l(gb):
    
    p = int(p_entry3.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)

    equation_str = equation_entry3.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v= solve_edo3(equation_str, gb)
    

    text = ""
    
    for i in range(0, p+1):
        text += f"[{i}] \ny1: {y_v[i]:.6f} \nz1: {z_v[i]:.6f}\nw1: {w_v[i]:.6f}\n\n"
    
    result_text3.delete('1.0', tk.END)
    result_text3.insert(tk.END, text)

def solve_edo3(equation_str, gb):
    global g
    g = gb
    # Obter os valores informados pelo usuário
    x0 = float(x0_entry3.get())
    x_final = float(x0_final_entry3.get())
    y0 = float(y0_entry3.get())
    z0 = float(z0_entry3.get())
    w0 = float(w0_entry3.get())
    n = int(np_entry3.get())
    h = (x_final - x0) / n
    derivative_str = equation_str

    # Definir a variável simbólica para x e y
    x, y, dydx, d2ydx2 = symbols('x y dydx d2ydx2')

    # Definir a função f(x, y) como a derivada de y (y')
    derivative_expr = sympify(derivative_str)
    f = lambdify((x, y, dydx, d2ydx2), derivative_expr, modules=['numpy'])

    # Resolvendo a equação usando o método de Runge-Kutta
    if gb == 4:
        x_values, y_values, z_values, w_values = runge_kutta_4th_order_edo_3th_order(f, x0, y0, z0, w0, x_final, n)
    elif gb == 6:
        x_values, y_values, z_values, w_values = runge_kutta_6th_order_edo_3th_order(f, x0, y0, z0, w0, x_final, n)
    elif gb == 1:
        x_values, y_values, z_values, w_values = solve_euler3(f, x0, y0, z0, w0, x_final, n)
    elif gb == 2:
        x_values, y_values, z_values, w_values = solve_heun3(f, x0, y0, z0, w0, x_final, n)

    return x_values, y_values, z_values, w_values

def solve_heun3(f, x0, y0, z0, w0, x_max, n):
    step = (x_max - x0) / n
    
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]
    w_values = [w0]

    x = x0
    y = y0
    z = z0
    w = w0

    for _ in range(n):
        k1 = step * z
        l1 = step * w
        m1 = step * f(x, y, z, w)

        k2 = step * (z + 0.5 * l1)
        l2 = step * (w + 0.5 * m1)
        m2 = step * f(x + 0.5 * step, y + 0.5 * k1, z + 0.5 * l1, w + 0.5 * m1)

        y_next = y + k2
        z_next = z + l2
        w_next = w + m2

        x = x + step
        y = y_next
        z = z_next
        w = w_next

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        w_values.append(w)

    return x_values, y_values, z_values, w_values

def solve_euler3(f, x0, y0, z0, w0, x_max, n):
    step = (x_max - x0) / n
    
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]
    w_values = [w0]

    x = x0
    y = y0
    z = z0
    w = w0

    for _ in range(n):
        y_next = y + step * z
        z_next = z + step * w
        w_next = w + step * f(x, y, z, w)

        x = x + step
        y = y_next
        z = z_next
        w = w_next

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        w_values.append(w)

    return x_values, y_values, z_values, w_values

def runge_kutta_4th_order_edo_3th_order(f, x0, y0, y_prime0, y_double_prime0, x_max, n):
    h = (x_max - x0) / n
     # Inicializar listas para armazenar os resultados
    x_values = [x0]
    y_values = [y0]
    y_prime_values = [y_prime0]
    y_double_prime_values = [y_double_prime0]

    # Implementação do método de Runge-Kutta de quarta ordem
    for _ in range(1, n+1):
        x = x_values[-1]
        y = y_values[-1]
        y_prime = y_prime_values[-1]
        y_double_prime = y_double_prime_values[-1]

        k1 = h * y_prime
        l1 = h * y_double_prime
        m1 = h * f(x, y, y_prime, y_double_prime)

        k2 = h * (y_prime + 0.5 * l1)
        l2 = h * (y_double_prime + 0.5 * m1)
        m2 = h * f(x + 0.5 * h, y + 0.5 * k1, y_prime + 0.5 * l1, y_double_prime + 0.5 * m1)

        k3 = h * (y_prime + 0.5 * l2)
        l3 = h * (y_double_prime + 0.5 * m2)
        m3 = h * f(x + 0.5 * h, y + 0.5 * k2, y_prime + 0.5 * l2, y_double_prime + 0.5 * m2)

        k4 = h * (y_prime + l3)
        l4 = h * (y_double_prime + m3)
        m4 = h * f(x + h, y + k3, y_prime + l3, y_double_prime + m3)

        x_values.append(x + h)
        y_values.append(y + (1/6) * (k1 + 2*k2 + 2*k3 + k4))
        y_prime_values.append(y_prime + (1/6) * (l1 + 2*l2 + 2*l3 + l4))
        y_double_prime_values.append(y_double_prime + (1/6) * (m1 + 2*m2 + 2*m3 + m4))

    return x_values, y_values, y_prime_values, y_double_prime_values

def runge_kutta_6th_order_edo_3th_order(f, x0, y0, y_prime0, y_double_prime0, x_max, n):
    h = (x_max - x0) / n
    x_values = np.linspace(x0, x_max, n + 1)
    y1_values = np.zeros(n + 1)
    y2_values = np.zeros(n + 1)
    y3_values = np.zeros(n + 1)

    x = x0
    y1 = y0
    y2 = y_prime0 
    y3 = y_double_prime0 

    for i in range(1, n + 1):
        k1_y1 = h * y2
        k1_y2 = h * y3
        k1_y3 = h * f(x, y1, y2, y3)

        k2_y1 = h * (y2 + k1_y2 / 3)
        k2_y2 = h * (y3 + k1_y3 / 3)
        k2_y3 = h * f(x + h / 3, y1 + k1_y1 / 3, y2 + k1_y2 / 3, y3 + k1_y3 / 3)

        k3_y1 = h * (y2 + 2 * k2_y2 / 3)
        k3_y2 = h * (y3 + 2 * k2_y3 / 3)
        k3_y3 = h * f(x + 2 * h / 3, y1 + 2 * k2_y1 / 3, y2 + 2 * k2_y2 / 3, y3 + 2 * k2_y3 / 3)

        k4_y1 = h * (y2 + k1_y2)
        k4_y2 = h * (y3 + k1_y3)
        k4_y3 = h * f(x + h, y1 + k1_y1, y2 + k1_y2, y3 + k1_y3)

        k5_y1 = h * (y2 + (k1_y2 + 4 * k2_y2 + k4_y2) / 6)
        k5_y2 = h * (y3 + (k1_y3 + 4 * k2_y3 + k4_y3) / 6)
        k5_y3 = h * f(x + h / 2, y1 + (k1_y1 + 4 * k2_y1 + k4_y1) / 6, y2 + (k1_y2 + 4 * k2_y2 + k4_y2) / 6, y3 + (k1_y3 + 4 * k2_y3 + k4_y3) / 6)

        k6_y1 = h * (y2 + (k1_y2 + 3 * k3_y2 + 4 * k4_y2 + k5_y2) / 8)
        k6_y2 = h * (y3 + (k1_y3 + 3 * k3_y3 + 4 * k4_y3 + k5_y3) / 8)
        k6_y3 = h * f(x + h, y1 + (k1_y1 + 3 * k3_y1 + 4 * k4_y1 + k5_y1) / 8, y2 + (k1_y2 + 3 * k3_y2 + 4 * k4_y2 + k5_y2) / 8, y3 + (k1_y3 + 3 * k3_y3 + 4 * k4_y3 + k5_y3) / 8)

        y1 = y1 + (k1_y1 + 3 * k3_y1 + 4 * k4_y1 + k5_y1 + 3 * k6_y1) / 15
        y2 = y2 + (k1_y2 + 3 * k3_y2 + 4 * k4_y2 + k5_y2 + 3 * k6_y2) / 15
        y3 = y3 + (k1_y3 + 3 * k3_y3 + 4 * k4_y3 + k5_y3 + 3 * k6_y3) / 15

        x = x + h
        x_values[i] = x
        y1_values[i] = y1
        y2_values[i] = y2
        y3_values[i] = y3

    return x_values, y1_values, y2_values, y3_values

# 4 ORDEM ###########

def plot_graph_fourth_order():
    x_values, y_values, _, _, _ = solve_edo4(equation_entry4.get(), g)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x")
    plt.legend()
    plt.grid(True)
    plt.show()

def iniciar_4nd_ordem_i(gb):
    p = int(p_entry4.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)
    d3ydx3 = Function('d3ydx3')(x)

    equation_str = equation_entry4.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v, j_v= solve_edo4(equation_str, gb)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label4.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def iniciar_4nd_ordem_l(gb):
    
    p = int(p_entry4.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)
    d3ydx3 = Function('d3ydx3')(x)

    equation_str = equation_entry4.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v, j_v = solve_edo4(equation_str, gb)
    

    text = ""
    
    for i in range(0, p+1):
        text += f"[{i}] \ny1: {y_v[i]:.6f} \nz1: {z_v[i]:.6f}\nw1: {w_v[i]:.6f}\nj1: {j_v[i]:.6f}\n\n"
    
    result_text4.delete('1.0', tk.END)
    result_text4.insert(tk.END, text)

def solve_edo4(equation_str, gb):
    global g
    g = gb
    # Obter os valores informados pelo usuário
    x0 = float(x0_entry4.get())
    x_final = float(x0_final_entry4.get())
    y0 = float(y0_entry4.get())
    z0 = float(z0_entry4.get())
    w0 = float(w0_entry4.get())
    j0 = float(j0_entry4.get())
    n = int(np_entry4.get())
    h = (x_final - x0) / n
    derivative_str = equation_str

    # Definir a variável simbólica para x e y
    x, y, dydx, d2ydx2, d3ydx3 = symbols('x y dydx d2ydx2 d3ydx3')

    # Definir a função f(x, y) como a derivada de y (y')
    derivative_expr = sympify(derivative_str)
    f = lambdify((x, y, dydx, d2ydx2, d3ydx3), derivative_expr, modules=['numpy'])

    # Resolvendo a equação usando o método de Runge-Kutta
    # Resolvendo a equação usando o método de Runge-Kutta
    if gb == 4:
        x_values, y_values, z_values, w_values, j_values = runge_kutta_4th_order_edo_4th_order(f, x0, y0, z0, w0, j0, x_final, n)
    elif gb == 6:
        x_values, y_values, z_values, w_values, j_values = runge_kutta_6th_order_edo_4th_order(f, x0, y0, z0, w0, j0, x_final, n)
    elif gb == 1:
        x_values, y_values, z_values, w_values, j_values = solve_euler4(f, x0, y0, z0, w0, j0, x_final, n)
    elif gb == 2:
        x_values, y_values, z_values, w_values, j_values = solve_heun4(f, x0, y0, z0, w0, j0, x_final, n)

    return x_values, y_values, z_values, w_values, j_values

def solve_euler4(f, x0, y0, z0, w0, j0, x_max, n):
    step = (x_max - x0) / n
    
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]
    w_values = [w0]
    j_values = [j0]

    x = x0
    y = y0
    z = z0
    w = w0
    j = j0

    for _ in range(n):
        y_next = y + step * z
        z_next = z + step * w
        w_next = w + step * j
        j_next = j + step * f(x, y, z, w, j)

        x = x + step
        y = y_next
        z = z_next
        w = w_next
        j = j_next

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        w_values.append(w)
        j_values.append(j)

    return x_values, y_values, z_values, w_values, j_values

def solve_heun4(f, x0, y0, z0, w0, j0, x_max, n):
    step = (x_max - x0) / n
    
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]
    w_values = [w0]
    j_values = [j0]

    x = x0
    y = y0
    z = z0
    w = w0
    j = j0

    for _ in range(n):
        k1 = step * z
        l1 = step * w
        m1 = step * j
        n1 = step * f(x, y, z, w, j)

        k2 = step * (z + 0.5 * l1)
        l2 = step * (w + 0.5 * m1)
        m2 = step * (j + 0.5 * n1)
        n2 = step * f(x + 0.5 * step, y + 0.5 * k1, z + 0.5 * l1, w + 0.5 * m1, j + 0.5 * n1)

        y_next = y + k2
        z_next = z + l2
        w_next = w + m2
        j_next = j + n2

        x = x + step
        y = y_next
        z = z_next
        w = w_next
        j = j_next

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        w_values.append(w)
        j_values.append(j)

    return x_values, y_values, z_values, w_values, j_values

def runge_kutta_4th_order_edo_4th_order(f, x0, y0, y_prime0, y_double_prime0, y_triple_prime0, x_max, n ):
    h = (x_max - x0) / n
    x_values = [x0]
    y_values = [y0]
    y_prime_values = [y_prime0]
    y_double_prime_values = [y_double_prime0]
    y_triple_prime_values = [y_triple_prime0]

    # Implementação do método de Runge-Kutta de quarta ordem
    for _ in range(1, n+1):
        x = x_values[-1]
        y = y_values[-1]
        y_prime = y_prime_values[-1]
        y_double_prime = y_double_prime_values[-1]
        y_triple_prime = y_triple_prime_values[-1]

        k1 = h * y_prime
        l1 = h * y_double_prime
        m1 = h * y_triple_prime
        n1 = h * f(x, y, y_prime, y_double_prime, y_triple_prime)

        k2 = h * (y_prime + 0.5 * l1)
        l2 = h * (y_double_prime + 0.5 * m1)
        m2 = h * (y_triple_prime + 0.5 * n1)
        n2 = h * f(x + 0.5 * h, y + 0.5 * k1, y_prime + 0.5 * l1, y_double_prime + 0.5 * m1, y_triple_prime + 0.5 * n1)

        k3 = h * (y_prime + 0.5 * l2)
        l3 = h * (y_double_prime + 0.5 * m2)
        m3 = h * (y_triple_prime + 0.5 * n2)
        n3 = h * f(x + 0.5 * h, y + 0.5 * k2, y_prime + 0.5 * l2, y_double_prime + 0.5 * m2, y_triple_prime + 0.5 * n2)

        k4 = h * (y_prime + l3)
        l4 = h * (y_double_prime + m3)
        m4 = h * (y_triple_prime + n3)
        n4 = h * f(x + h, y + k3, y_prime + l3, y_double_prime + m3, y_triple_prime + n3)

        x_values.append(x + h)
        y_values.append(y + (1/6) * (k1 + 2*k2 + 2*k3 + k4))
        y_prime_values.append(y_prime + (1/6) * (l1 + 2*l2 + 2*l3 + l4))
        y_double_prime_values.append(y_double_prime + (1/6) * (m1 + 2*m2 + 2*m3 + m4))
        y_triple_prime_values.append(y_triple_prime + (1/6) * (n1 + 2*n2 + 2*n3 + n4))

    return x_values, y_values, y_prime_values, y_double_prime_values, y_triple_prime_values

def runge_kutta_6th_order_edo_4th_order(f, x0, y0, y_prime0, y_double_prime0, y_triple_prime0, x_max, n):
    h = (x_max - x0) / n
    x_values = np.linspace(x0, x_max, n + 1)
    y1_values = np.zeros(n + 1)
    y2_values = np.zeros(n + 1)
    y3_values = np.zeros(n + 1)
    y4_values = np.zeros(n + 1)

    x = x0
    y1 = y0
    y2 = y_prime0  
    y3 = y_double_prime0  
    y4 = y_triple_prime0  

    for i in range(1, n + 1):
        k1_y1 = h * y2
        k1_y2 = h * y3
        k1_y3 = h * y4
        k1_y4 = h * f(x, y1, y2, y3, y4)

        k2_y1 = h * (y2 + k1_y2 / 3)
        k2_y2 = h * (y3 + k1_y3 / 3)
        k2_y3 = h * (y4 + k1_y4 / 3)
        k2_y4 = h * f(x + h / 3, y1 + k1_y1 / 3, y2 + k1_y2 / 3, y3 + k1_y3 / 3, y4 + k1_y4 / 3)

        k3_y1 = h * (y2 + k2_y2 / 3)
        k3_y2 = h * (y3 + k2_y3 / 3)
        k3_y3 = h * (y4 + k2_y4 / 3)
        k3_y4 = h * f(x + 2 * h / 3, y1 + k2_y1 / 3, y2 + k2_y2 / 3, y3 + k2_y3 / 3, y4 + k2_y4 / 3)

        k4_y1 = h * (y2 + k3_y2)
        k4_y2 = h * (y3 + k3_y3)
        k4_y3 = h * (y4 + k3_y4)
        k4_y4 = h * f(x + h, y1 + k3_y1, y2 + k3_y2, y3 + k3_y3, y4 + k3_y4)

        k5_y1 = h * (y2 + (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 7)
        k5_y2 = h * (y3 + (k1_y3 + 2 * k2_y3 + 2 * k3_y3 + k4_y3) / 7)
        k5_y3 = h * (y4 + (k1_y4 + 2 * k2_y4 + 2 * k3_y4 + k4_y4) / 7)
        k5_y4 = h * f(x + 5 * h / 6, y1 + (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1) / 7, y2 + (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 7, y3 + (k1_y3 + 2 * k2_y3 + 2 * k3_y3 + k4_y3) / 7, y4 + (k1_y4 + 2 * k2_y4 + 2 * k3_y4 + k4_y4) / 7)

        k6_y1 = h * (y2 + (7 * k1_y2 + 10 * k2_y2 + k4_y2) / 27)
        k6_y2 = h * (y3 + (7 * k1_y3 + 10 * k2_y3 + k4_y3) / 27)
        k6_y3 = h * (y4 + (7 * k1_y4 + 10 * k2_y4 + k4_y4) / 27)
        k6_y4 = h * f(x + h, y1 + (7 * k1_y1 + 10 * k2_y1 + k4_y1) / 27, y2 + (7 * k1_y2 + 10 * k2_y2 + k4_y2) / 27, y3 + (7 * k1_y3 + 10 * k2_y3 + k4_y3) / 27, y4 + (7 * k1_y4 + 10 * k2_y4 + k4_y4) / 27)

        y1 = y1 + (k1_y1 + 3 * k3_y1 + 4 * k4_y1 + k5_y1 + 3 * k6_y1) / 15
        y2 = y2 + (k1_y2 + 3 * k3_y2 + 4 * k4_y2 + k5_y2 + 3 * k6_y2) / 15
        y3 = y3 + (k1_y3 + 3 * k3_y3 + 4 * k4_y3 + k5_y3 + 3 * k6_y3) / 15
        y4 = y4 + (k1_y4 + 3 * k3_y4 + 4 * k4_y4 + k5_y4 + 3 * k6_y4) / 15

        x = x + h
        x_values[i] = x
        y1_values[i] = y1
        y2_values[i] = y2
        y3_values[i] = y3
        y4_values[i] = y4

    return x_values, y1_values, y2_values, y3_values, y4_values

# 5 ORDEM ###########

def plot_graph_fifth_order():
    x_values, y_values, _, _, _, _ = solve_edo5(equation_entry5.get(), g)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x")
    plt.legend()
    plt.grid(True)
    plt.show()

def iniciar_5nd_ordem_i(gb):
    p = int(p_entry5.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)
    d3ydx3 = Function('d3ydx3')(x)
    d4ydx4 = Function('d4ydx4')(x)

    equation_str = equation_entry5.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v, j_v, c_v= solve_edo5(equation_str, gb)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label5.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def iniciar_5nd_ordem_l(bg):
    
    p = int(p_entry5.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)
    d3ydx3 = Function('d3ydx3')(x)
    d4ydx4 = Function('d4ydx4')(x)

    equation_str = equation_entry5.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v, j_v, c_v = solve_edo5(equation_str, gb)
    

    text = ""
    
    for i in range(0, p+1):
        text += f"[{i}] \ny1: {y_v[i]:.6f} \nz1: {z_v[i]:.6f}\nw1: {w_v[i]:.6f}\nj1: {j_v[i]:.6f}\nj1: {c_v[i]:.6f}\n\n"
    
    result_text5.delete('1.0', tk.END)
    result_text5.insert(tk.END, text)

def solve_edo5(equation_str, gb):
    global g
    g = gb
    # Obter os valores informados pelo usuário
    x0 = float(x0_entry5.get())
    x_final = float(x0_final_entry5.get())
    y0 = float(y0_entry5.get())
    z0 = float(z0_entry5.get())
    w0 = float(w0_entry5.get())
    j0 = float(j0_entry5.get())
    c0 = float(c0_entry5.get())
    n = int(np_entry5.get())
    h = (x_final - x0) / n
    derivative_str = equation_str

    # Definir a variável simbólica para x e y
    x, y, dydx, d2ydx2, d3ydx3, d4ydx4 = symbols('x y dydx d2ydx2 d3ydx3 d4ydx4')

    # Definir a função f(x, y) como a derivada de y (y')
    derivative_expr = sympify(derivative_str)
    f = lambdify((x, y, dydx, d2ydx2, d3ydx3, d4ydx4), derivative_expr, modules=['numpy'])

    if gb == 4:
        x_values, y_values, z_values, w_values, j_values, c_values = runge_kutta_4th_order_edo_5th_order(f, x0, y0, z0, w0, j0, c0, x_final, n)
    elif gb == 6:
        x_values, y_values, z_values, w_values, j_values, c_values = runge_kutta_6th_order_edo_5th_order(f, x0, y0, z0, w0, j0, c0, x_final, n)
    elif gb == 1:
        x_values, y_values, z_values, w_values, j_values, c_values = solve_euler5(f, x0, y0, z0, w0, j0, c0, x_final, n)
    elif gb == 2:
        x_values, y_values, z_values, w_values, j_value, c_valuess = solve_heun5(f, x0, y0, z0, w0, j0, c0, x_final, n)

    return x_values, y_values, z_values, w_values, j_values, c_values

def solve_euler5(f, x0, y0, z0, w0, j0, c0, x_max, n):
    step = (x_max - x0) / n
    
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]
    w_values = [w0]
    j_values = [j0]
    c_values = [c0]

    x = x0
    y = y0
    z = z0
    w = w0
    j = j0
    c = c0

    for _ in range(n):
        y_next = y + step * z
        z_next = z + step * w
        w_next = w + step * j
        j_next = j + step * c
        c_next = c + step * f(x, y, z, w, j, c)

        x = x + step
        y = y_next
        z = z_next
        w = w_next
        j = j_next
        c = c_next

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        w_values.append(w)
        j_values.append(j)
        c_values.append(c)

    return x_values, y_values, z_values, w_values, j_values, c_values

def solve_heun5(f, x0, y0, z0, w0, j0, c0, x_max, n):
    step = (x_max - x0) / n
    
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]
    w_values = [w0]
    j_values = [j0]
    c_values = [c0]

    x = x0
    y = y0
    z = z0
    w = w0
    j = j0
    c = c0

    for _ in range(n):
        k1_y = step * z
        k1_z = step * w
        k1_w = step * j
        k1_j = step * c
        k1_c = step * f(x, y, z, w, j, c)

        k2_y = step * (z + k1_z)
        k2_z = step * (w + k1_w)
        k2_w = step * (j + k1_j)
        k2_j = step * (c + k1_c)
        k2_c = step * f(x + step, y + k1_y, z + k1_z, w + k1_w, j + k1_j, c + k1_c)

        y_next = y + 0.5 * (k1_y + k2_y)
        z_next = z + 0.5 * (k1_z + k2_z)
        w_next = w + 0.5 * (k1_w + k2_w)
        j_next = j + 0.5 * (k1_j + k2_j)
        c_next = c + 0.5 * (k1_c + k2_c)

        x = x + step
        y = y_next
        z = z_next
        w = w_next
        j = j_next
        c = c_next

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        w_values.append(w)
        j_values.append(j)
        c_values.append(c)

    return x_values, y_values, z_values, w_values, j_values, c_values

def runge_kutta_4th_order_edo_5th_order(f, x0, y0, y_prime0, y_double_prime0, y_triple_prime0, y_quadruple_prime0, x_max, n):
    h = (x_max - x0) / n
    
    x_values = [x0]
    y_values = [y0]
    y_prime_values = [y_prime0]
    y_double_prime_values = [y_double_prime0]
    y_triple_prime_values = [y_triple_prime0]
    y_quadruple_prime_values = [y_quadruple_prime0]

    # Implementação do método de Runge-Kutta de quinta ordem
    for _ in range(1, n+1):
        x = x_values[-1]
        y = y_values[-1]
        y_prime = y_prime_values[-1]
        y_double_prime = y_double_prime_values[-1]
        y_triple_prime = y_triple_prime_values[-1]
        y_quadruple_prime = y_quadruple_prime_values[-1]

        k1 = h * y_prime
        l1 = h * y_double_prime
        m1 = h * y_triple_prime
        n1 = h * y_quadruple_prime
        p1 = h * f(x, y, y_prime, y_double_prime, y_triple_prime, y_quadruple_prime)

        k2 = h * (y_prime + 0.5 * l1)
        l2 = h * (y_double_prime + 0.5 * m1)
        m2 = h * (y_triple_prime + 0.5 * n1)
        n2 = h * (y_quadruple_prime + 0.5 * p1)
        p2 = h * f(x + 0.5 * h, y + 0.5 * k1, y_prime + 0.5 * l1, y_double_prime + 0.5 * m1, y_triple_prime + 0.5 * n1, y_quadruple_prime + 0.5 * p1)

        k3 = h * (y_prime + 0.5 * l2)
        l3 = h * (y_double_prime + 0.5 * m2)
        m3 = h * (y_triple_prime + 0.5 * n2)
        n3 = h * (y_quadruple_prime + 0.5 * p2)
        p3 = h * f(x + 0.5 * h, y + 0.5 * k2, y_prime + 0.5 * l2, y_double_prime + 0.5 * m2, y_triple_prime + 0.5 * n2, y_quadruple_prime + 0.5 * p2)

        k4 = h * (y_prime + l3)
        l4 = h * (y_double_prime + m3)
        m4 = h * (y_triple_prime + n3)
        n4 = h * (y_quadruple_prime + p3)
        p4 = h * f(x + h, y + k3, y_prime + l3, y_double_prime + m3, y_triple_prime + n3, y_quadruple_prime + p3)

        x_values.append(x + h)
        y_values.append(y + (1/6) * (k1 + 2*k2 + 2*k3 + k4))
        y_prime_values.append(y_prime + (1/6) * (l1 + 2*l2 + 2*l3 + l4))
        y_double_prime_values.append(y_double_prime + (1/6) * (m1 + 2*m2 + 2*m3 + m4))
        y_triple_prime_values.append(y_triple_prime + (1/6) * (n1 + 2*n2 + 2*n3 + n4))
        y_quadruple_prime_values.append(y_quadruple_prime + (1/6) * (p1 + 2*p2 + 2*p3 + p4))

    return x_values, y_values, y_prime_values, y_double_prime_values, y_triple_prime_values, y_quadruple_prime_values

def runge_kutta_6th_order_edo_5th_order(f, x0, y0, y_prime0, y_double_prime0, y_triple_prime0, y_quadruple_prime0, x_max, n):
    h = (x_max - x0) / n
    x_values = np.linspace(x0, x_max, n + 1)
    y1_values = np.zeros(n + 1)
    y2_values = np.zeros(n + 1)
    y3_values = np.zeros(n + 1)
    y4_values = np.zeros(n + 1)
    y5_values = np.zeros(n + 1)

    x = x0
    y1 = y0
    y2 = y_prime0  
    y3 = y_double_prime0  
    y4 = y_triple_prime0  
    y5 = y_quadruple_prime0  

    for i in range(1, n + 1):
        k1_y1 = h * y2
        k1_y2 = h * y3
        k1_y3 = h * y4
        k1_y4 = h * y5
        k1_y5 = h * f(x, y1, y2, y3, y4, y5)

        k2_y1 = h * (y2 + k1_y2 / 3)
        k2_y2 = h * (y3 + k1_y3 / 3)
        k2_y3 = h * (y4 + k1_y4 / 3)
        k2_y4 = h * (y5 + k1_y5 / 3)
        k2_y5 = h * f(x + h / 3, y1 + k1_y1 / 3, y2 + k1_y2 / 3, y3 + k1_y3 / 3, y4 + k1_y4 / 3, y5 + k1_y5 / 3)

        k3_y1 = h * (y2 + k2_y2 / 3)
        k3_y2 = h * (y3 + k2_y3 / 3)
        k3_y3 = h * (y4 + k2_y4 / 3)
        k3_y4 = h * (y5 + k2_y5 / 3)
        k3_y5 = h * f(x + 2 * h / 3, y1 + k2_y1 / 3, y2 + k2_y2 / 3, y3 + k2_y3 / 3, y4 + k2_y4 / 3, y5 + k2_y5 / 3)

        k4_y1 = h * (y2 + k3_y2)
        k4_y2 = h * (y3 + k3_y3)
        k4_y3 = h * (y4 + k3_y4)
        k4_y4 = h * (y5 + k3_y5)
        k4_y5 = h * f(x + h, y1 + k3_y1, y2 + k3_y2, y3 + k3_y3, y4 + k3_y4, y5 + k3_y5)

        k5_y1 = h * (y2 + (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 7)
        k5_y2 = h * (y3 + (k1_y3 + 2 * k2_y3 + 2 * k3_y3 + k4_y3) / 7)
        k5_y3 = h * (y4 + (k1_y4 + 2 * k2_y4 + 2 * k3_y4 + k4_y4) / 7)
        k5_y4 = h * (y5 + (k1_y5 + 2 * k2_y5 + 2 * k3_y5 + k4_y5) / 7)
        k5_y5 = h * f(x + 5 * h / 6, y1 + (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1) / 7, y2 + (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 7, y3 + (k1_y3 + 2 * k2_y3 + 2 * k3_y3 + k4_y3) / 7, y4 + (k1_y4 + 2 * k2_y4 + 2 * k3_y4 + k4_y4) / 7, y5 + (k1_y5 + 2 * k2_y5 + 2 * k3_y5 + k4_y5) / 7)

        k6_y1 = h * (y2 + (7 * k1_y2 + 10 * k2_y2 + k4_y2) / 27)
        k6_y2 = h * (y3 + (7 * k1_y3 + 10 * k2_y3 + k4_y3) / 27)
        k6_y3 = h * (y4 + (7 * k1_y4 + 10 * k2_y4 + k4_y4) / 27)
        k6_y4 = h * (y5 + (7 * k1_y5 + 10 * k2_y5 + k4_y5) / 27)
        k6_y5 = h * f(x + h, y1 + (7 * k1_y1 + 10 * k2_y1 + k4_y1) / 27, y2 + (7 * k1_y2 + 10 * k2_y2 + k4_y2) / 27, y3 + (7 * k1_y3 + 10 * k2_y3 + k4_y3) / 27, y4 + (7 * k1_y4 + 10 * k2_y4 + k4_y4) / 27, y5 + (7 * k1_y5 + 10 * k2_y5 + k4_y5) / 27)

        y1 = y1 + (k1_y1 + 3 * k3_y1 + 4 * k4_y1 + k5_y1 + 3 * k6_y1) / 15
        y2 = y2 + (k1_y2 + 3 * k3_y2 + 4 * k4_y2 + k5_y2 + 3 * k6_y2) / 15
        y3 = y3 + (k1_y3 + 3 * k3_y3 + 4 * k4_y3 + k5_y3 + 3 * k6_y3) / 15
        y4 = y4 + (k1_y4 + 3 * k3_y4 + 4 * k4_y4 + k5_y4 + 3 * k6_y4) / 15
        y5 = y5 + (k1_y5 + 3 * k3_y5 + 4 * k4_y5 + k5_y5 + 3 * k6_y5) / 15

        x = x + h
        x_values[i] = x
        y1_values[i] = y1
        y2_values[i] = y2
        y3_values[i] = y3
        y4_values[i] = y4
        y5_values[i] = y5

    return x_values, y1_values, y2_values, y3_values, y4_values, y5_values


##Parte Grafica

# Criar a janela principal
root = tk.Tk()
root.title("Resolver EDO's de 1nd-5nd Ordem")

root.geometry("430x500")

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

# Criar um nootbok (aba)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)


# Primeira aba
def on_configure1(event):
    canvas1.configure(scrollregion=canvas1.bbox("all"))
    

tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='1° Ordem')  

# Criando Canvas
canvas1 = tk.Canvas(tab1)
canvas1.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Criando Scroll
scrollbar1_1 = ttk.Scrollbar(tab1, orient=tk.VERTICAL, command=canvas1.yview)
scrollbar1_1.pack(side=tk.RIGHT, fill=tk.Y)

canvas1.configure(yscrollcommand=scrollbar1_1.set )
canvas1.bind('<Configure>',on_configure1)

# Criando frame dentro do canvas
frame_aux1 = tk.Frame(canvas1)

canvas1.create_window((0,0), window=frame_aux1, anchor = "nw" )

# Valores pedidos ao usuario
equation_label1 = tk.Label(frame_aux1, text="Digite a derivada de y (y'): ")
equation_label1.pack(pady=(10, 0), padx=(120,120))

equation_entry1 = tk.Entry(frame_aux1)
equation_entry1.pack()


x0_label1 = tk.Label(frame_aux1, text="Digite o valor inicial de x (x0): ")
x0_label1.pack()

x0_entry1 = tk.Entry(frame_aux1)
x0_entry1.pack()

x0_final_label1 = tk.Label(frame_aux1, text="Digite o valor final de x (xf): ")
x0_final_label1.pack()

x0_final_entry1 = tk.Entry(frame_aux1)
x0_final_entry1.pack()

y0_label1 = tk.Label(frame_aux1, text="Digite o valor inicial de y (y0): ")
y0_label1.pack()

y0_entry1 = tk.Entry(frame_aux1)
y0_entry1.pack()

n_label1 = tk.Label(frame_aux1, text="Digite a quantidade de pontos (np): ")
n_label1.pack()

np_entry1 = tk.Entry(frame_aux1)
np_entry1.pack()

p_label1 = tk.Label(frame_aux1, text="Digite o ponto que deseja (p): ")
p_label1.pack()

p_entry1 = tk.Entry(frame_aux1)
p_entry1.pack()

g = 0

buttons_frame_1 = tk.Frame(frame_aux1)
buttons_frame_1.pack()

# Botão para calcular com rk4
solve_button1_1 = tk.Button(buttons_frame_1, width=8, text="RK4", command=lambda: iniciar_1nd_ordem_i(4), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1_1.grid(row=0, column=0, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button1_2 = tk.Button(buttons_frame_1, width=8, text="RK6", command=lambda: iniciar_1nd_ordem_i(6), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1_2.grid(row=0, column=1, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid

# Botão para calcular com rk4
solve_button1_3 = tk.Button(buttons_frame_1, width=8, text="EL1", command=lambda: iniciar_1nd_ordem_i(1), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1_3.grid(row=1, column=0, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button1_4 = tk.Button(buttons_frame_1, width=8, text="EL2", command=lambda: iniciar_1nd_ordem_i(2), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1_4.grid(row=1, column=1, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid


# Botão para listar y1
solve_button1_5 = tk.Button(frame_aux1, width=12, text="Listar y1", command=lambda: iniciar_1nd_ordem_l(g), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1_5.pack(pady=(10, 0))

# Botaão para plotar y/x
plot_button_1 = tk.Button(frame_aux1, width=12, text="Plotar Gráfico", command=plot_graph_first_order, bd=2, bg='#107db2', fg='white',
                        font=('verdana', 8, 'bold'))
plot_button_1.pack(pady=(10, 0))


# Rótulo para mostrar o resultado final de y1
result_label1 = tk.Label(frame_aux1, text="")
result_label1.pack(pady=(10, 5))

# Lista de valores
result_frame1 = tk.Frame(frame_aux1)
result_frame1.pack()

result_text1 = tk.Text(result_frame1, wrap=tk.WORD, width=25, height=7.5)
result_text1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Use grid

scrollbar1_2 = tk.Scrollbar(result_frame1, command=result_text1.yview)
scrollbar1_2.grid(row=0, column=1, sticky="ns")  # Use grid

result_text1.config(yscrollcommand=scrollbar1_2.set)

#######################

# Segunda aba
def on_configure2(event):
    canvas2.configure(scrollregion=canvas2.bbox("all"))
    

tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='2° Ordem')

# Criando Canvas
canvas2 = tk.Canvas(tab2)
canvas2.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Criando Scroll
scrollbar2_1 = ttk.Scrollbar(tab2, orient=tk.VERTICAL, command=canvas2.yview)
scrollbar2_1.pack(side=tk.RIGHT, fill=tk.Y)

canvas2.configure(yscrollcommand=scrollbar2_1.set )
canvas2.bind('<Configure>',on_configure2)

# Criando frame dentro do canvas
frame_aux2 = tk.Frame(canvas2)

canvas2.create_window((0,0), window=frame_aux2, anchor = "nw" )


# Valores pedidos ao usuario
equation_label2 = tk.Label(frame_aux2, text="Digite a segunda derivada de y (y''): ")
equation_label2.pack(pady=(10,0), padx=(100,100))

equation_entry2 = tk.Entry(frame_aux2)
equation_entry2.pack()

x0_inicial_label2 = tk.Label(frame_aux2, text="Digite o valor inicial de x, (x0): ")
x0_inicial_label2.pack()

x0_entry2 = tk.Entry(frame_aux2)
x0_entry2.pack()

x0_final_label2 = tk.Label(frame_aux2, text="Digite o valor final de x, (xf): ")
x0_final_label2.pack()

x0_final_entry2 = tk.Entry(frame_aux2)
x0_final_entry2.pack()

y0_label2 = tk.Label(frame_aux2, text="Digite o valor inicial de y, y(x0): ")
y0_label2.pack()

y0_entry2 = tk.Entry(frame_aux2)
y0_entry2.pack()

z0_label2 = tk.Label(frame_aux2, text="Digite o valor inicial de z, y'(x0): ")
z0_label2.pack()

z0_entry2 = tk.Entry(frame_aux2)
z0_entry2.pack()

n_label2 = tk.Label(frame_aux2, text="numero de pontos (n): ")
n_label2.pack()

n_entry2 = tk.Entry(frame_aux2)
n_entry2.pack()

p_label2 = tk.Label(frame_aux2, text="ponto desejado (p): ")
p_label2.pack()

p_entry2 = tk.Entry(frame_aux2)
p_entry2.pack()

buttons_frame_2 = tk.Frame(frame_aux2)
buttons_frame_2.pack()

# Botão para calcular com rk4
solve_button2_1 = tk.Button(buttons_frame_2, width=8, text="RK4", command=lambda: iniciar_2nd_ordem_i(4), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2_1.grid(row=0, column=0, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button2_2 = tk.Button(buttons_frame_2, width=8, text="RK6", command=lambda: iniciar_2nd_ordem_i(6), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2_2.grid(row=0, column=1, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid

# Botão para calcular com rk4
solve_button2_3 = tk.Button(buttons_frame_2, width=8, text="EL1", command=lambda: iniciar_2nd_ordem_i(1), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2_3.grid(row=1, column=0, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button2_4 = tk.Button(buttons_frame_2, width=8, text="EL2", command=lambda: iniciar_2nd_ordem_i(2), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2_4.grid(row=1, column=1, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid

# Botão para listar y1
solve_button2_5 = tk.Button(frame_aux2, width=12, text="Listar y1", command=lambda: iniciar_2nd_ordem_l(g), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2_5.pack(pady=(10, 0))

# Botão para plotar grafico 
plot_button_2 = tk.Button(frame_aux2, width=12, text="Plotar Gráfico", command=plot_graph_second_order, bd=2, bg='#107db2',
                                      fg='white', font=('verdana', 8, 'bold'))
plot_button_2.pack(pady=(10, 0))


# Rótulo para mostrar o resultado final de y1
result_label2 = tk.Label(frame_aux2, text="")
result_label2.pack(pady=(10, 0))


# Lista de valores
result_frame2 = tk.Frame(frame_aux2)
result_frame2.pack()

result_text2 = tk.Text(result_frame2, wrap=tk.WORD, width=25, height=7.5)
result_text2.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Use grid

scrollbar2 = tk.Scrollbar(result_frame2, command=result_text2.yview)
scrollbar2.grid(row=0, column=1, sticky="ns")  # Use grid

result_text2.config(yscrollcommand=scrollbar2.set)

##############

# Terceira aba

def on_configure3(event):
    canvas3.configure(scrollregion=canvas3.bbox("all"))
    
# Criando aba
tab3 = ttk.Frame(notebook)
notebook.add(tab3, text='3° Ordem')  

# Criando Canvas
canvas3 = tk.Canvas(tab3)
canvas3.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Criando Scroll
scrollbar3_1 = ttk.Scrollbar(tab3, orient=tk.VERTICAL, command=canvas3.yview)
scrollbar3_1.pack(side=tk.RIGHT, fill=tk.Y)

canvas3.configure(yscrollcommand=scrollbar3_1.set )
canvas3.bind('<Configure>',on_configure3)

# Criando frame dentro do canvas
frame_aux3 = tk.Frame(canvas3)

canvas3.create_window((0,0), window=frame_aux3, anchor = "nw" )

# Valores pedidos ao usuario
equation_label3 = tk.Label(frame_aux3, text="Digite a terceira derivada de y (y'''): ")
equation_label3.pack(pady=(10,0), padx=(100,100))

equation_entry3 = tk.Entry(frame_aux3)
equation_entry3.pack(pady=(5,0))


x0_label3 = tk.Label(frame_aux3, text="Digite o valor inicial de x (x0): ")
x0_label3.pack(pady=(5,0))


x0_entry3 = tk.Entry(frame_aux3)
x0_entry3.pack(pady=(5,0))


x0_final_label3 = tk.Label(frame_aux3, text="Digite o valor final de x (xf): ")
x0_final_label3.pack(pady=(5,0))

x0_final_entry3 = tk.Entry(frame_aux3)
x0_final_entry3.pack(pady=(5,0))

y0_label3 = tk.Label(frame_aux3, text="Digite o valor inicial de y (y0): ")
y0_label3.pack(pady=(5,0))

y0_entry3 = tk.Entry(frame_aux3)
y0_entry3.pack(pady=(5,0))

z0_label3 = tk.Label(frame_aux3, text="Digite o valor inicial de z, y'(x0): ")
z0_label3.pack(pady=(5,0))

z0_entry3 = tk.Entry(frame_aux3)
z0_entry3.pack(pady=(5,0))

w0_label3 = tk.Label(frame_aux3, text="Digite o valor inicial de w, y''(x0): ")
w0_label3.pack(pady=(5,0))

w0_entry3 = tk.Entry(frame_aux3)
w0_entry3.pack(pady=(5,0))

n_label3 = tk.Label(frame_aux3, text="Digite a quantidade de pontos (np): ")
n_label3.pack(pady=(5,0))

np_entry3 = tk.Entry(frame_aux3)
np_entry3.pack(pady=(5,0))

p_label3 = tk.Label(frame_aux3, text="Digite o ponto que deseja (p): ")
p_label3.pack(pady=(5,0))

p_entry3 = tk.Entry(frame_aux3)
p_entry3.pack(pady=(5,0))

buttons_frame_3 = tk.Frame(frame_aux3)
buttons_frame_3.pack()

# Botão para calcular com rk4
solve_button3_1 = tk.Button(buttons_frame_3, width=8, text="RK4", command=lambda: iniciar_3nd_ordem_i(4), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button3_1.grid(row=0, column=0, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button3_2 = tk.Button(buttons_frame_3, width=8, text="RK6", command=lambda: iniciar_3nd_ordem_i(6), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button3_2.grid(row=0, column=1, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid

# Botão para calcular com rk4
solve_button3_3 = tk.Button(buttons_frame_3, width=8, text="EL1", command=lambda: iniciar_3nd_ordem_i(1), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button3_3.grid(row=1, column=0, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button3_4 = tk.Button(buttons_frame_3, width=8, text="EL2", command=lambda: iniciar_3nd_ordem_i(2), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button3_4.grid(row=1, column=1, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid

# Botão para listar y1
solve_button3_5 = tk.Button(frame_aux3, width=12, text="Listar y1", command=lambda: iniciar_3nd_ordem_l(g), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button3_5.pack(pady=(10, 0))

# Botão para plotar grafico 
plot_button_3 = tk.Button(frame_aux3, width=12, text="Plotar Gráfico", command=plot_graph_third_order, bd=2, bg='#107db2',
                                      fg='white', font=('verdana', 8, 'bold'))
plot_button_3.pack(pady=(10, 0))


# Rótulo para mostrar o resultado final de y1
result_label3 = tk.Label(frame_aux3, text="")
result_label3.pack(pady=(5,0))

# Lista de valores
result_frame3 = tk.Frame(frame_aux3)
result_frame3.pack(pady=(5,0))

result_text3 = tk.Text(result_frame3, wrap=tk.WORD, width=25, height=7.5)
result_text3.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Use grid

scrollbar3_2 = tk.Scrollbar(result_frame3, command=result_text3.yview)
scrollbar3_2.grid(row=0, column=1, sticky="ns")  # Use grid

result_text3.config(yscrollcommand=scrollbar3_2.set)

# Quarta aba

def on_configure4(event):
    canvas4.configure(scrollregion=canvas4.bbox("all"))
    
# Criando aba
tab4 = ttk.Frame(notebook)
notebook.add(tab4, text='4° Ordem')  

# Criando Canvas
canvas4 = tk.Canvas(tab4)
canvas4.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Criando Scroll
scrollbar4_1 = ttk.Scrollbar(tab4, orient=tk.VERTICAL, command=canvas4.yview)
scrollbar4_1.pack(side=tk.RIGHT, fill=tk.Y)

canvas4.configure(yscrollcommand=scrollbar4_1.set )
canvas4.bind('<Configure>',on_configure4)

# Criando frame dentro do canvas
frame_aux4 = tk.Frame(canvas4)

canvas4.create_window((0,0), window=frame_aux4, anchor = "nw" )

# Valores pedidos ao usuario
equation_label4 = tk.Label(frame_aux4, text="Digite a quarta derivada de y (y''''): ")
equation_label4.pack(pady=(10,0), padx=(100,100))

equation_entry4 = tk.Entry(frame_aux4)
equation_entry4.pack(pady=(5,0))


x0_label4 = tk.Label(frame_aux4, text="Digite o valor inicial de x (x0): ")
x0_label4.pack(pady=(5,0))


x0_entry4 = tk.Entry(frame_aux4)
x0_entry4.pack(pady=(5,0))


x0_final_label4 = tk.Label(frame_aux4, text="Digite o valor final de x (xf): ")
x0_final_label4.pack(pady=(5,0))

x0_final_entry4 = tk.Entry(frame_aux4)
x0_final_entry4.pack(pady=(5,0))

y0_label4 = tk.Label(frame_aux4, text="Digite o valor inicial de y (y0): ")
y0_label4.pack(pady=(5,0))

y0_entry4 = tk.Entry(frame_aux4)
y0_entry4.pack(pady=(5,0))

z0_label4 = tk.Label(frame_aux4, text="Digite o valor inicial de z, y'(x0): ")
z0_label4.pack(pady=(5,0))

z0_entry4 = tk.Entry(frame_aux4)
z0_entry4.pack(pady=(5,0))

w0_label4 = tk.Label(frame_aux4, text="Digite o valor inicial de w, y''(x0): ")
w0_label4.pack(pady=(5,0))

w0_entry4 = tk.Entry(frame_aux4)
w0_entry4.pack(pady=(5,0))

j0_label4 = tk.Label(frame_aux4, text="Digite o valor inicial de j, y'''(x0): ")
j0_label4.pack(pady=(5,0))

j0_entry4 = tk.Entry(frame_aux4)
j0_entry4.pack(pady=(5,0))

n_label4 = tk.Label(frame_aux4, text="Digite a quantidade de pontos (np): ")
n_label4.pack(pady=(5,0))

np_entry4 = tk.Entry(frame_aux4)
np_entry4.pack(pady=(5,0))

p_label4 = tk.Label(frame_aux4, text="Digite o ponto que deseja (p): ")
p_label4.pack(pady=(5,0))

p_entry4 = tk.Entry(frame_aux4)
p_entry4.pack(pady=(5,0))


buttons_frame_4 = tk.Frame(frame_aux4)
buttons_frame_4.pack()

# Botão para calcular com rk4
solve_button4_1 = tk.Button(buttons_frame_4, width=8, text="RK4", command=lambda: iniciar_4nd_ordem_i(4), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button4_1.grid(row=0, column=0, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button4_2 = tk.Button(buttons_frame_4, width=8, text="RK6", command=lambda: iniciar_4nd_ordem_i(6), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button4_2.grid(row=0, column=1, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid

# Botão para calcular com rk4
solve_button4_3 = tk.Button(buttons_frame_4, width=8, text="EL1", command=lambda: iniciar_4nd_ordem_i(1), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button4_3.grid(row=1, column=0, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button4_4 = tk.Button(buttons_frame_4, width=8, text="EL2", command=lambda: iniciar_4nd_ordem_i(2), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button4_4.grid(row=1, column=1, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid

# Botão para listar y1
solve_button4_5 = tk.Button(frame_aux4, width=12, text="Listar y1", command=lambda: iniciar_4nd_ordem_l(g), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button4_5.pack(pady=(10, 0))

# Botão para plotar grafico 
plot_button_4 = tk.Button(frame_aux4, width=12, text="Plotar Gráfico", command=plot_graph_fourth_order, bd=2, bg='#107db2',
                                      fg='white', font=('verdana', 8, 'bold'))
plot_button_4.pack(pady=(10, 0))

# Rótulo para mostrar o resultado final de y1
result_label4 = tk.Label(frame_aux4, text="")
result_label4.pack(pady=(5,0))

# Lista de valores
result_frame4 = tk.Frame(frame_aux4)
result_frame4.pack(pady=(5,0))

result_text4 = tk.Text(result_frame4, wrap=tk.WORD, width=25, height=7.5)
result_text4.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Use grid

scrollbar4_2 = tk.Scrollbar(result_frame4, command=result_text4.yview)
scrollbar4_2.grid(row=0, column=1, sticky="ns")  # Use grid

result_text4.config(yscrollcommand=scrollbar4_2.set)

# Quinta aba

def on_configure5(event):
    canvas5.configure(scrollregion=canvas5.bbox("all"))
    
# Criando aba
tab5 = ttk.Frame(notebook)
notebook.add(tab5, text='5° Ordem')  

# Criando Canvas
canvas5 = tk.Canvas(tab5)
canvas5.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Criando Scroll
scrollbar5_1 = ttk.Scrollbar(tab5, orient=tk.VERTICAL, command=canvas5.yview)
scrollbar5_1.pack(side=tk.RIGHT, fill=tk.Y)

canvas5.configure(yscrollcommand=scrollbar5_1.set )
canvas5.bind('<Configure>',on_configure5)

# Criando frame dentro do canvas
frame_aux5 = tk.Frame(canvas5)

canvas5.create_window((0,0), window=frame_aux5, anchor = "nw" )

# Valores pedidos ao usuario
equation_label5 = tk.Label(frame_aux5, text="Digite a quarta derivada de y (y''''): ")
equation_label5.pack(pady=(10,0), padx=(100,100))

equation_entry5 = tk.Entry(frame_aux5)
equation_entry5.pack(pady=(5,0))


x0_label5 = tk.Label(frame_aux5, text="Digite o valor inicial de x (x0): ")
x0_label5.pack(pady=(5,0))


x0_entry5 = tk.Entry(frame_aux5)
x0_entry5.pack(pady=(5,0))


x0_final_label5 = tk.Label(frame_aux5, text="Digite o valor final de x (xf): ")
x0_final_label5.pack(pady=(5,0))

x0_final_entry5 = tk.Entry(frame_aux5)
x0_final_entry5.pack(pady=(5,0))

y0_label5 = tk.Label(frame_aux5, text="Digite o valor inicial de y (y0): ")
y0_label5.pack(pady=(5,0))

y0_entry5 = tk.Entry(frame_aux5)
y0_entry5.pack(pady=(5,0))

z0_label5 = tk.Label(frame_aux5, text="Digite o valor inicial de z, y'(x0): ")
z0_label5.pack(pady=(5,0))

z0_entry5 = tk.Entry(frame_aux5)
z0_entry5.pack(pady=(5,0))

w0_label5 = tk.Label(frame_aux5, text="Digite o valor inicial de w, y''(x0): ")
w0_label5.pack(pady=(5,0))

w0_entry5 = tk.Entry(frame_aux5)
w0_entry5.pack(pady=(5,0))

j0_label5 = tk.Label(frame_aux5, text="Digite o valor inicial de j, y'''(x0): ")
j0_label5.pack(pady=(5,0))

j0_entry5 = tk.Entry(frame_aux5)
j0_entry5.pack(pady=(5,0))

c0_label5 = tk.Label(frame_aux5, text="Digite o valor inicial de c, y'''(x0): ")
c0_label5.pack(pady=(5,0))

c0_entry5 = tk.Entry(frame_aux5)
c0_entry5.pack(pady=(5,0))

n_label5 = tk.Label(frame_aux5, text="Digite a quantidade de pontos (np): ")
n_label5.pack(pady=(5,0))

np_entry5 = tk.Entry(frame_aux5)
np_entry5.pack(pady=(5,0))

p_label5 = tk.Label(frame_aux5, text="Digite o ponto que deseja (p): ")
p_label5.pack(pady=(5,0))

p_entry5 = tk.Entry(frame_aux5)
p_entry5.pack(pady=(5,0))

buttons_frame_5 = tk.Frame(frame_aux5)
buttons_frame_5.pack()

# Botão para calcular com rk5
solve_button5_1 = tk.Button(buttons_frame_5, width=8, text="RK4", command=lambda: iniciar_5nd_ordem_i(4), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button5_1.grid(row=0, column=0, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button5_2 = tk.Button(buttons_frame_5, width=8, text="RK6", command=lambda: iniciar_5nd_ordem_i(6), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button5_2.grid(row=0, column=1, padx=(10, 0), pady=(10, 10), sticky="nsew")  # Use grid

# Botão para calcular com rk5
solve_button5_3 = tk.Button(buttons_frame_5, width=8, text="EL1", command=lambda: iniciar_5nd_ordem_i(1), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button5_3.grid(row=1, column=0, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid


# Botão para calcular com rk6
solve_button5_4 = tk.Button(buttons_frame_5, width=8, text="EL2", command=lambda: iniciar_5nd_ordem_i(2), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button5_4.grid(row=1, column=1, padx=(10, 0), pady=(0, 10), sticky="nsew")  # Use grid

# Botão para listar y1
solve_button5_5 = tk.Button(frame_aux5, width=12, text="Listar y1", command=lambda: iniciar_5nd_ordem_l(g), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button5_5.pack(pady=(10, 0))

# Botão para plotar grafico 
plot_button_5 = tk.Button(frame_aux5, width=12, text="Plotar Gráfico", command=plot_graph_fifth_order, bd=2, bg='#107db2',
                                      fg='white', font=('verdana', 8, 'bold'))
plot_button_5.pack(pady=(10, 0))



# Rótulo para mostrar o resultado final de y1
result_label5 = tk.Label(frame_aux5, text="")
result_label5.pack(pady=(5,0))

# Lista de valores
result_frame5 = tk.Frame(frame_aux5)
result_frame5.pack(pady=(5,0))

result_text5 = tk.Text(result_frame5, wrap=tk.WORD, width=25, height=7.5)
result_text5.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Use grid

scrollbar5_2 = tk.Scrollbar(result_frame5, command=result_text4.yview)
scrollbar5_2.grid(row=0, column=1, sticky="ns")  # Use grid

result_text5.config(yscrollcommand=scrollbar5_2.set)


root.mainloop()
