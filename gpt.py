import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

def inciar_3nd_ordem_i():
    global x_sym, y_sym, dydx_sym, dy2dx_sym, equation

    # Solicitar informações do usuário
    
    x0 = float(x0_entry3.get())
    x_final = float(x0_final_entry3.get())
    y0 = float(y0_entry3.get())
    z0 = float(z0_entry3.get())
    w0 = float(w0_entry3.get())
    n = int(np_entry3.get())
    p = int(p_entry3.get())

    #Definir simbolos
    x_sym = Symbol('x')
    y_sym = Function('y')(x_sym)
    dydx_sym = Function('dydx')(x_sym)
    dy2dx_sym = Function('dy2dx')(x_sym)

    equation = equation_entry3.get()
    equation = sympify(equation)

    y_v, z_v, w_v = solve_edo3(x0, x_final, y0, z0, w0, n, p)

    result_label3.config(text=f"y1 = {y_v[p]:.6f}, z1 = {z_v[p]:.6f}, w1 = {w_v[p]:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def system_of_odes3(x, yzw):
    y, z, w = yzw
    dydx = z
    dzdx = w
    dwdx = eval(str(lambdify((x_sym, y_sym, dydx_sym, dy2dx_sym), equation)(x, y, w, dzdx)))
    return [dydx, dzdx, dwdx]

def solve_edo3(x0, x_final, y0, z0, w0, n, p):

    global x_values
    global y_values
    global z_values
    global w_values

    # Definir o intervalo de integração
    x_span = (x0, x_final)

    # Definir as condições iniciais
    initial_conditions = [y0, z0, w0]

    # Resolver o sistema usando o método de Runge-Kutta de 4ª ordem
    solution = solve_ivp(system_of_odes3, x_span, initial_conditions, method='RK45', t_eval=np.linspace(x0, x_final, n+1))

    # Obter os resultados
    x_values = solution.t
    y_values, z_values, w_values = solution.y

    return y_values, z_values, w_values

##Parte Grafica

# Criar a janela principal
root = tk.Tk()
root.title("Resolver EDO's de 1nd e 2nd Ordem")

root.geometry("300x600")

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

# Criar um nootbok (aba)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

def on_configure(event):
    canvas3.configure(scrollregion=canvas3.bbox("all"))
    
# Criando aba
tab3 = ttk.Frame(notebook)
notebook.add(tab3, text='3° Ordem')  

# Criando Canvas
canvas3 = tk.Canvas(tab3)
canvas3.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Criando Scroll
scroll = ttk.Scrollbar(tab3, orient=tk.VERTICAL, command=canvas3.yview)
scroll.pack(side=tk.RIGHT, fill=tk.Y)

canvas3.configure(yscrollcommand=scroll.set )
canvas3.bind('<Configure>',on_configure)

# Criando frame dentro do canvas
frame_aux = tk.Frame(canvas3)

canvas3.create_window((0,0), window=frame_aux, anchor = "nw" )

# Valores pedidos ao usuario
equation_label3 = tk.Label(frame_aux, text="Digite a derivada de y (y'): ")
equation_label3.pack(pady=(10,0))

equation_entry3 = tk.Entry(frame_aux)
equation_entry3.pack(pady=(5,0))


x0_label3 = tk.Label(frame_aux, text="Digite o valor inicial de x (x0): ")
x0_label3.pack(pady=(5,0))


x0_entry3 = tk.Entry(frame_aux)
x0_entry3.pack(pady=(5,0))


x0_final_label3 = tk.Label(frame_aux, text="Digite o valor final de x (xf): ")
x0_final_label3.pack(pady=(5,0))

x0_final_entry3 = tk.Entry(frame_aux)
x0_final_entry3.pack(pady=(5,0))

y0_label3 = tk.Label(frame_aux, text="Digite o valor inicial de y (y0): ")
y0_label3.pack(pady=(5,0))

y0_entry3 = tk.Entry(frame_aux)
y0_entry3.pack(pady=(5,0))

z0_label3 = tk.Label(frame_aux, text="Digite o valor inicial de z, y'(x0): ")
z0_label3.pack(pady=(5,0))

z0_entry3 = tk.Entry(frame_aux)
z0_entry3.pack(pady=(5,0))

w0_label3 = tk.Label(frame_aux, text="Digite o valor inicial de w, y''(x0): ")
w0_label3.pack(pady=(5,0))

w0_entry3 = tk.Entry(frame_aux)
w0_entry3.pack(pady=(5,0))

n_label3 = tk.Label(frame_aux, text="Digite a quantidade de pontos (np): ")
n_label3.pack(pady=(5,0))

np_entry3 = tk.Entry(frame_aux)
np_entry3.pack(pady=(5,0))

p_label3 = tk.Label(frame_aux, text="Digite o ponto que deseja (p): ")
p_label3.pack(pady=(5,0))

p_entry3 = tk.Entry(frame_aux)
p_entry3.pack(pady=(5,0))

# Botão para calcular y1
solve_button5 = tk.Button(frame_aux, text="Calcular y1", command=inciar_3nd_ordem_i, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button5.pack(pady=(5,0))

# Rótulo para mostrar o resultado final de y1
result_label3 = tk.Label(frame_aux, text="")
result_label3.pack(pady=(5,0))

root.mainloop()
