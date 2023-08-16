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
        x_values, y_values = solve_pvi_rk4th_edo_1th_order(f, x0, y0, x_final, n)
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

def solve_pvi_rk4th_edo_1th_order(f, x0, y0, x_final, n):
    
    # Use solve_ivp to solve the IVP
    sol = solve_ivp(f, (x0, x_final), [y0], t_eval=np.linspace(x0, x_final, n+1), method='RK45')

    return sol.t, sol.y[0]

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

root.mainloop()
######################