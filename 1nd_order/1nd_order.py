import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
import tkinter as tk
from tkinter import ttk

def convert_to_first_orderi(gb):
    p = int(p_entry1.get())
    equation_str = derivative_entry1.get()
    x = Symbol('x')
    y = Function('y')(x)

    eq1 = eval(equation_str)

    x_v, y_v = solve_runge_kutta(equation_str, gb)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label1.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def convert_to_first_orderl(gb):
    p = int(p_entry1.get())
    equation_str = derivative_entry1.get()
    x = Symbol('x')
    y = Function('y')(x)

    # Convertendo a string da equação para a forma simbólica
    equation_str = equation_str.replace("y''", "y'")
    equation_str = equation_str.replace("y'", "y")
    eq1 = eval(equation_str)

    x_v, y_v = solve_runge_kutta(equation_str, gb)

    text = ""
    
    for i in range(0, p+1):
        text += f"{[i]} y1: {y_v[i]:.6f}\n"
    
    result_text1.delete('1.0', tk.END)
    result_text1.insert(tk.END, text)
   
def solve_runge_kutta(equation_str, gb):
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
    if (g == 4):
        x_values, y_values = runge_kutta_4th_order(f, x0, y0, x_final, n)
    elif (g == 6):
        x_values, y_values = runge_kutta_6th_order(f, x0, y0, x_final, n)
    
    return x_values, y_values

def runge_kutta_4th_order(f, x0, y0, x_max, n):
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

def runge_kutta_6th_order(f, x0, y0, x_max, n):
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

#############

##Parte Grafica
g = 0
# Criar a janela principal
root = tk.Tk()
root.title("Resolver EDO's de 1nd Ordem")

root.geometry("300x575")

# Criar um nootbok (aba)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)


# Primeira aba
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='1° Ordem')  

# Valores pedidos ao usuario
derivative_label1 = tk.Label(tab1, text="Digite a derivada de y (y'): ")
derivative_label1.pack(pady=(10, 0))

derivative_entry1 = tk.Entry(tab1)
derivative_entry1.pack()


x0_label1 = tk.Label(tab1, text="Digite o valor inicial de x (x0): ")
x0_label1.pack()

x0_entry1 = tk.Entry(tab1)
x0_entry1.pack()

x0_final_label1 = tk.Label(tab1, text="Digite o valor final de x (xf): ")
x0_final_label1.pack()

x0_final_entry1 = tk.Entry(tab1)
x0_final_entry1.pack()

y0_label1 = tk.Label(tab1, text="Digite o valor inicial de y (y0): ")
y0_label1.pack()

y0_entry1 = tk.Entry(tab1)
y0_entry1.pack()

n_label1 = tk.Label(tab1, text="Digite a quantidade de pontos (np): ")
n_label1.pack()

np_entry1 = tk.Entry(tab1)
np_entry1.pack()

p_label1 = tk.Label(tab1, text="Digite o ponto que deseja (p): ")
p_label1.pack()

p_entry1 = tk.Entry(tab1)
p_entry1.pack()

# Botão para calcular y1
solve_button1 = tk.Button(tab1, text="RK4", command=lambda: convert_to_first_orderi(4), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1.pack(pady=(40, 0))

# Botão para calcular y1
solve_button1 = tk.Button(tab1, text="RK6", command=lambda: convert_to_first_orderi(6), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1.pack(pady=(10, 0))


# Botão para calcular y1
solve_button2 = tk.Button(tab1, text="Listar y1", command=lambda:convert_to_first_orderl(g), bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2.pack(pady=(10, 0))


# Rótulo para mostrar o resultado final de y1
result_label1 = tk.Label(tab1, text="")
result_label1.pack(pady=(10, 10))

# Lista de valores
result_frame1 = tk.Frame(tab1)
result_frame1.pack()

result_text1 = tk.Text(result_frame1, wrap=tk.WORD, width=25, height=7.5)
result_text1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Use grid

scrollbar1 = tk.Scrollbar(result_frame1, command=result_text1.yview)
scrollbar1.grid(row=0, column=1, sticky="ns")  # Use grid

result_text1.config(yscrollcommand=scrollbar1.set)

#######################

root.mainloop()


