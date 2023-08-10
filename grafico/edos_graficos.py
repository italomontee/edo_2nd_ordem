import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

def plot_graph():
    x_values, y_values = solve_runge_kutta(derivative_entry1.get())
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x")
    plt.legend()
    plt.grid(True)
    plt.show()


def convert_to_first_orderi():
    p = int(p_entry1.get())
    equation_str = derivative_entry1.get()
    x = Symbol('x')
    y = Function('y')(x)

    # Convertendo a string da equação para a forma simbólica
    equation_str = equation_str.replace("y''", "y'")
    equation_str = equation_str.replace("y'", "y")
    eq1 = eval(equation_str)

    x_v, y_v = solve_runge_kutta(equation_str)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label1.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def convert_to_first_orderl():
    p = int(p_entry1.get())
    equation_str = derivative_entry1.get()
    x = Symbol('x')
    y = Function('y')(x)

    # Convertendo a string da equação para a forma simbólica
    equation_str = equation_str.replace("y''", "y'")
    equation_str = equation_str.replace("y'", "y")
    eq1 = eval(equation_str)

    x_v, y_v = solve_runge_kutta(equation_str)

    text = ""
    
    for i in range(0, p+1):
        text += f"{[i]} y1: {y_v[i]:.6f}\n"
    
    result_text1.delete('1.0', tk.END)
    result_text1.insert(tk.END, text)

    
def solve_runge_kutta(equation_str):
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
    x_values, y_values = runge_kutta_4th_order(f, x0, y0, x_final, n)

    
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

#############
def plot_graph_second_order():
    y_values, _ = solve_edo(float(x0_inicial_entry2.get()), float(x0_final_entry2.get()),
                             float(y0_entry2.get()), float(z0_entry2.get()), int(n_entry2.get()), int(p_entry2.get()))

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label="y(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gráfico de y em função de x (2ª Ordem)")
    plt.legend()
    plt.grid(True)
    plt.show()


#Função para iniciar o processo
def iniciari():
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

    x_v, y_v = solve_edo(x0, x_final, y0, z0, n, p)

    result_label3.config(text=f"y1 = {y_values[p]:.6f}, z1 = {z_values[p]:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def iniciarl():
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

    y_v, z_v = solve_edo(x0, x_final, y0, z0, n, p)

    text = ""
    
    for i in range(0, p+1):
        text += f"[{i}] \ny1: {y_v[i]:.6f} \nz1: {z_v[i]:.6f}\n\n"
    
    result_text2.delete('1.0', tk.END)
    result_text2.insert(tk.END, text)                  

# Função que representa o sistema de EDOs
def system_of_odes(x, y_z):
    y, z = y_z
    dydx = z
    dzdx = eval(str(lambdify((x_sym, y_sym, dydx_sym), equation)(x, y, dydx)))
    return [dydx, dzdx]

#Função para resolver o usando rk45
def solve_edo(x0, x_final, y0, z0, n, p):

    global x_values
    global y_values
    global z_values

    # Definir o intervalo de integração
    x_span = (x0, x_final)

    # Definir as condições iniciais
    initial_conditions = [y0, z0]

    # Resolver o sistema usando o método de Runge-Kutta de 4ª ordem
    solution = solve_ivp(system_of_odes, x_span, initial_conditions, method='RK45', t_eval=np.linspace(x0, x_final, n+1))

    # Obter os resultados
    x_values = solution.t
    y_values, z_values = solution.y

    return y_values, z_values

    #result_label2.config(text=f"y1 = {y_values[p]:.6f}, z1 = {z_values[p]:.6f}",  bd=2, bg = '#107db2', fg ='white'
    #                        , font = ('verdana', 8, 'bold'))

#####################

##Parte Grafica

# Criar a janela principal
root = tk.Tk()
root.title("Resolver EDO's de 1nd e 2nd Ordem")

root.geometry("300x600")

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
solve_button1 = tk.Button(tab1, text="Calcular y1", command=convert_to_first_orderi, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1.pack(pady=(20, 0))

# Botão para calcular y1
solve_button2 = tk.Button(tab1, text="Listar y1", command=convert_to_first_orderl, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2.pack(pady=(10, 0))

# Botaão para plotar y/x
plot_button = tk.Button(tab1, text="Plotar Gráfico", command=plot_graph, bd=2, bg='#107db2', fg='white',
                        font=('verdana', 8, 'bold'))
plot_button.pack(pady=(10, 0))


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

# Segunda aba
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='2° Ordem')

# Valores pedidos ao usuario
equation_label2 = tk.Label(tab2, text="Entre com a EDO (use x, y e dydx)")
equation_label2.pack(pady=(10,0))

equation_entry2 = tk.Entry(tab2)
equation_entry2.pack()

x0_inicial_label2 = tk.Label(tab2, text="Digite o valor inicial de x, (x0): ")
x0_inicial_label2.pack()

x0_inicial_entry2 = tk.Entry(tab2)
x0_inicial_entry2.pack()

x0_final_label2 = tk.Label(tab2, text="Digite o valor final de x, (xf): ")
x0_final_label2.pack()

x0_final_entry2 = tk.Entry(tab2)
x0_final_entry2.pack()

y0_label2 = tk.Label(tab2, text="Digite o valor inicial de y, y(x0): ")
y0_label2.pack()

y0_entry2 = tk.Entry(tab2)
y0_entry2.pack()

z0_label2 = tk.Label(tab2, text="Digite o valor inicial de z, y'(x0): ")
z0_label2.pack()

z0_entry2 = tk.Entry(tab2)
z0_entry2.pack()

n_label2 = tk.Label(tab2, text="numero de pontos (n): ")
n_label2.pack()

n_entry2 = tk.Entry(tab2)
n_entry2.pack()

p_label2 = tk.Label(tab2, text="ponto desejado (p): ")
p_label2.pack()

p_entry2 = tk.Entry(tab2)
p_entry2.pack()

# Botão para calcular y1
solve_button3 = tk.Button(tab2, text="Calcular y1 e z1", command=iniciari, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button3.pack(pady=(10, 0))

# Botão para listar y1
solve_button4 = tk.Button(tab2, text="Listar y1 e z1", command=iniciarl, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button4.pack(pady=(10, 0))

# Botão para plotar grafico 
plot_button_second_order = tk.Button(tab2, text="Plotar Gráfico", command=plot_graph_second_order, bd=2, bg='#107db2',
                                      fg='white', font=('verdana', 8, 'bold'))
plot_button_second_order.pack(pady=(10, 0))


# Rótulo para mostrar o resultado final de y1
result_label3 = tk.Label(tab2, text="")
result_label3.pack(pady=(10, 0))


# Lista de valores
result_frame2 = tk.Frame(tab2)
result_frame2.pack()

result_text2 = tk.Text(result_frame2, wrap=tk.WORD, width=25, height=7.5)
result_text2.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Use grid

scrollbar2 = tk.Scrollbar(result_frame2, command=result_text2.yview)
scrollbar2.grid(row=0, column=1, sticky="ns")  # Use grid

result_text2.config(yscrollcommand=scrollbar2.set)
##############


root.mainloop()


# Imprimir os resultados
#print("\nResultados:")
#for i in range(len(x_values)):
#    print(f"x = {x_values[i]:.2f}, y = {y_values[i]:.6f}, z = {z_values[i]:.6f}")
