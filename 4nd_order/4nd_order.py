import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# 1 ORDEM ############

def plot_graph_first_order():
    x_values, y_values = solve_edo1(equation_entry1.get())
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x")
    plt.legend()
    plt.grid(True)
    plt.show()

def inciar_1nd_ordem_i():
    p = int(p_entry1.get())
    
    x = Symbol('x')
    y = Function('y')(x)

    equation_str = equation_entry1.get()

    eq1 = eval(equation_str)

    x_v, y_v = solve_edo1(equation_str)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label1.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def inciar_1nd_ordem_l():
    p = int(p_entry1.get())

    x = Symbol('x')
    y = Function('y')(x)

    equation_str = equation_entry1.get()

    eq1 = eval(equation_str)

    x_v, y_v = solve_edo1(equation_str)

    text = ""
    
    for i in range(0, p+1):
        text += f"{[i]} y1: {y_v[i]:.6f}\n"
    
    result_text1.delete('1.0', tk.END)
    result_text1.insert(tk.END, text)
    
def solve_edo1(equation_str):
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
    x_values, y_values = runge_kutta_4th_order_edo_1th_order(f, x0, y0, x_final, n)

    
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

# 2 ORDEM ############

def plot_graph_second_order():
    y_values, _ = solve_edo2(float(x0_inicial_entry2.get()), float(x0_final_entry2.get()),
                             float(y0_entry2.get()), float(z0_entry2.get()), int(n_entry2.get()), int(p_entry2.get()))

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x (2ª Ordem)")
    plt.legend()
    plt.grid(True)
    plt.show()

def inciar_2nd_ordem_i():
    global x_sym, y_sym, dydx_sym, equation

    # Solicitar informações do usuário
    p = int(p_entry2.get())
    x0 = float(x0_inicial_entry2.get())
    x_final = float(x0_final_entry2.get())
    y0 = float(y0_entry2.get())
    z0 = float(z0_entry2.get())
    n = int(n_entry2.get())

    #Definir simbolos
    x_sym = Symbol('x')
    y_sym = Function('y')(x_sym)
    dydx_sym = Function('dydx')(x_sym)

    equation = equation_entry2.get()
    equation = sympify(equation)

    y_v, z_v = solve_edo2(x0, x_final, y0, z0, n, p)

    result_label2.config(text=f"y1 = {y_v[p]:.6f}, z1 = {z_v[p]:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def inciar_2nd_ordem_l():
    global x_sym, y_sym, dydx_sym, equation

    # Solicitar informações do usuário
    p = int(p_entry2.get())
    x0 = float(x0_inicial_entry2.get())
    x_final = float(x0_final_entry2.get())
    y0 = float(y0_entry2.get())
    z0 = float(z0_entry2.get())
    n = int(n_entry2.get())

    #Definir simbolos
    x_sym = Symbol('x')
    y_sym = Function('y')(x_sym)
    dydx_sym = Function('dydx')(x_sym)

    equation = equation_entry2.get()
    equation = sympify(equation)

    y_v, z_v = solve_edo2(x0, x_final, y0, z0, n, p)

    text = ""
    
    for i in range(0, p+1):
        text += f"[{i}] \ny1: {y_v[i]:.6f} \nz1: {z_v[i]:.6f}\n\n"
    
    result_text2.delete('1.0', tk.END)
    result_text2.insert(tk.END, text)                  

def system_of_odes2(x, y_z):
    y, z = y_z
    dydx = z
    dzdx = eval(str(lambdify((x_sym, y_sym, dydx_sym), equation)(x, y, dydx)))
    return [dydx, dzdx]

def solve_edo2(x0, x_final, y0, z0, n, p):

    global x_values
    global y_values
    global z_values

    # Definir o intervalo de integração
    x_span = (x0, x_final)

    # Definir as condições iniciais
    initial_conditions = [y0, z0]

    # Resolver o sistema usando o método de Runge-Kutta de 4ª ordem
    solution = solve_ivp(system_of_odes2, x_span, initial_conditions, method='RK45', t_eval=np.linspace(x0, x_final, n+1))

    # Obter os resultados
    x_values = solution.t
    y_values, z_values = solution.y

    return y_values, z_values

    #result_label2.config(text=f"y1 = {y_values[p]:.6f}, z1 = {z_values[p]:.6f}",  bd=2, bg = '#107db2', fg ='white'
    #                        , font = ('verdana', 8, 'bold'))

# 3 ORDEM ############

def plot_graph_third_order():
    x_values, y_values, _, _ = solve_edo3(equation_entry3.get())
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x")
    plt.legend()
    plt.grid(True)
    plt.show()

def inciar_3nd_ordem_i():
    
    p = int(p_entry3.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)

    equation_str = equation_entry3.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v= solve_edo3(equation_str)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label3.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def inciar_3nd_ordem_l():
    
    p = int(p_entry3.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)

    equation_str = equation_entry3.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v= solve_edo3(equation_str)
    

    text = ""
    
    for i in range(0, p+1):
        text += f"[{i}] \ny1: {y_v[i]:.6f} \nz1: {z_v[i]:.6f}\nw1: {w_v[i]:.6f}\n\n"
    
    result_text3.delete('1.0', tk.END)
    result_text3.insert(tk.END, text)

def solve_edo3(equation_str):

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
    x_values, y_values, z_values, w_values = runge_kutta_4th_order_edo_3th_order(f, x0, y0, z0, w0, h, n)
  
    return x_values, y_values, z_values, w_values

def runge_kutta_4th_order_edo_3th_order(f, x0, y0, y_prime0, y_double_prime0, h, num_points):
     # Inicializar listas para armazenar os resultados
    x_values = [x0]
    y_values = [y0]
    y_prime_values = [y_prime0]
    y_double_prime_values = [y_double_prime0]

    # Implementação do método de Runge-Kutta de quarta ordem
    for _ in range(num_points):
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

# 4 ORDEM ###########

def plot_graph_fourth_order():
    x_values, y_values, _, _, _ = solve_edo4(equation_entry4.get())
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x")
    plt.legend()
    plt.grid(True)
    plt.show()

def inciar_4nd_ordem_i():
    p = int(p_entry4.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)
    d3ydx3 = Function('d3ydx3')(x)

    equation_str = equation_entry4.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v, j_v= solve_edo4(equation_str)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label4.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def inciar_4nd_ordem_l():
    
    p = int(p_entry4.get())
    
    x = Symbol('x')
    y = Function('y')(x)
    dydx = Function('dydxy')(x)
    d2ydx2 = Function('d2ydx2')(x)
    d3ydx3 = Function('d3ydx3')(x)

    equation_str = equation_entry4.get()

    eq1 = eval(equation_str)

    x_v, y_v, z_v, w_v, j_v = solve_edo4(equation_str)
    

    text = ""
    
    for i in range(0, p+1):
        text += f"[{i}] \ny1: {y_v[i]:.6f} \nz1: {z_v[i]:.6f}\nw1: {w_v[i]:.6f}\nj1: {j_v[i]:.6f}\n\n"
    
    result_text4.delete('1.0', tk.END)
    result_text4.insert(tk.END, text)

def solve_edo4(equation_str):

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
    x_values, y_values, z_values, w_values, j_values = runge_kutta_4th_order_edo_4th_order(f, x0, y0, z0, w0, j0, h, n)
  
    return x_values, y_values, z_values, w_values, j_values

def runge_kutta_4th_order_edo_4th_order(f, x0, y0, y_prime0, y_double_prime0, y_triple_prime0, h, num_points):
    x_values = [x0]
    y_values = [y0]
    y_prime_values = [y_prime0]
    y_double_prime_values = [y_double_prime0]
    y_triple_prime_values = [y_triple_prime0]

    # Implementação do método de Runge-Kutta de quarta ordem
    for _ in range(num_points):
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

##Parte Grafica

# Criar a janela principal
root = tk.Tk()
root.title("Resolver EDO's de 1nd e 2nd Ordem")

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

# Botão para calcular y1
solve_button1_1 = tk.Button(frame_aux1, text="Calcular y1", command=inciar_1nd_ordem_i, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1_1.pack(pady=(20, 0))

# Botão para calcular y1
solve_button1_2 = tk.Button(frame_aux1, text="Listar y1", command=inciar_1nd_ordem_l, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1_2.pack(pady=(10, 0))

# Botaão para plotar y/x
plot_button_1 = tk.Button(frame_aux1, text="Plotar Gráfico", command=plot_graph_first_order, bd=2, bg='#107db2', fg='white',
                        font=('verdana', 8, 'bold'))
plot_button_1.pack(pady=(10, 0))


# Rótulo para mostrar o resultado final de y1
result_label1 = tk.Label(frame_aux1, text="")
result_label1.pack(pady=(10, 10))

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

x0_inicial_entry2 = tk.Entry(frame_aux2)
x0_inicial_entry2.pack()

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

# Botão para calcular y1
solve_button2_1 = tk.Button(frame_aux2, text="Calcular y1 e z1", command=inciar_2nd_ordem_i, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2_1.pack(pady=(10, 0))

# Botão para listar y1
solve_button2_2 = tk.Button(frame_aux2, text="Listar y1 e z1", command=inciar_2nd_ordem_l, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2_2.pack(pady=(10, 0))

# Botão para plotar grafico 
plot_button_2 = tk.Button(frame_aux2, text="Plotar Gráfico", command=plot_graph_second_order, bd=2, bg='#107db2',
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

# Botão para calcular y1
solve_button3_1 = tk.Button(frame_aux3, text="Calcular y1", command=inciar_3nd_ordem_i, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button3_1.pack(pady=(5,0))


#Botão para plotar y/x
# Botão para plotar grafico 
solve_button3_2 = tk.Button(frame_aux3, text="Listar y1, z1, w1", command=inciar_3nd_ordem_l, bd=2, bg='#107db2',
                                      fg='white', font=('verdana', 8, 'bold'))
solve_button3_2.pack(pady=(10, 0))
# Botão para plotar grafico 

plot_button_3 = tk.Button(frame_aux3, text="Plotar Gráfico", command=plot_graph_third_order, bd=2, bg='#107db2',
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


# Botão para calcular y1
solve_button4_1 = tk.Button(frame_aux4, text="Calcular y1", command=inciar_4nd_ordem_i, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button4_1.pack(pady=(5,0))


#Botão para plotar y/x
# Botão para plotar grafico 
solve_button4_2 = tk.Button(frame_aux4, text="Listar y1, z1, w1, j1", command=inciar_4nd_ordem_l, bd=2, bg='#107db2',
                                      fg='white', font=('verdana', 8, 'bold'))
solve_button4_2.pack(pady=(10, 0))
# Botão para plotar grafico 

plot_button_4 = tk.Button(frame_aux4, text="Plotar Gráfico", command=plot_graph_fourth_order, bd=2, bg='#107db2',
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


root.mainloop()
