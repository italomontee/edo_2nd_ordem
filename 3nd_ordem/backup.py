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

    x_1 = Symbol('x')
    y_1 = Function('y')(x_1)

    equation_str = equation_entry1.get()

    # Convertendo a string da equação para a forma simbólica
    equation_str = equation_str.replace("y''", "y'")
    equation_str = equation_str.replace("y'", "y")
    eq1 = eval(equation_str)

    x_v, y_v = solve_edo1(equation_str)
    
    y1 = y_v[p]

    # Atualizar o rótulo com o valor de y1
    result_label1.config(text=f"y1 = {y1:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def inciar_1nd_ordem_l():
    p = int(p_entry1.get())
    
    x_1 = Symbol('x')
    y_1 = Function('y')(x_1)

    equation_str = equation_entry1.get()

    # Convertendo a string da equação para a forma simbólica
    equation_str = equation_str.replace("y''", "y'")
    equation_str = equation_str.replace("y'", "y")
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
    x_1, y_1 = symbols('x y')

    # Definir a função f(x, y) como a derivada de y (y')
    derivative_expr = sympify(derivative_str)
    f = lambdify((x_1, y_1), derivative_expr, modules=['numpy'])

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
    global x_2_sym, y_2_sym, dydx_2_sym, equation_2

    # Solicitar informações do usuário
    p = int(p_entry2.get())
    x0 = float(x0_inicial_entry2.get())
    x_final = float(x0_final_entry2.get())
    y0 = float(y0_entry2.get())
    z0 = float(z0_entry2.get())
    n = int(n_entry2.get())

    #Definir simbolos
    x_2_sym = Symbol('x')
    y_2_sym = Function('y')(x_sym)
    dydx_2_sym = Function('dydx')(x_sym)

    equation_2 = equation_entry2.get()
    equation_2 = sympify(equation_2)

    y_v, z_v = solve_edo2(x0, x_final, y0, z0, n, p)

    result_label2.config(text=f"y1 = {y_v[p]:.6f}, z1 = {z_v[p]:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def inciar_2nd_ordem_l():
    global x_2_sym, y_2_sym, dydx_2_sym, equation_2

    # Solicitar informações do usuário
    p = int(p_entry2.get())
    x0 = float(x0_inicial_entry2.get())
    x_final = float(x0_final_entry2.get())
    y0 = float(y0_entry2.get())
    z0 = float(z0_entry2.get())
    n = int(n_entry2.get())

    #Definir simbolos
    x_2_sym = Symbol('x')
    y_2_sym = Function('y')(x_sym)
    dydx_2_sym = Function('dydx')(x_sym)

    equation_2 = equation_entry2.get()
    equation_2 = sympify(equation_2)

    y_v, z_v = solve_edo2(x0, x_final, y0, z0, n, p)

    text = ""
    
    for i in range(0, p+1):
        text += f"[{i}] \ny1: {y_v[i]:.6f} \nz1: {z_v[i]:.6f}\n\n"
    
    result_text2.delete('1.0', tk.END)
    result_text2.insert(tk.END, text)                  

def system_of_odes2(x, y_z):
    y, z = y_z
    dydx = z
    dzdx = eval(str(lambdify((x_2_sym, y_2_sym, dydx_2_sym), equation_2)(x, y, dydx)))
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

def inciar_3nd_ordem_i():
    global x_3_sym, y_3_sym, dydx_3_sym, dy2dx_3_sym, equation_3


    # Solicitar informações do usuário
    
    x0 = float(x0_entry3.get())
    x_final = float(x0_final_entry3.get())
    y0 = float(y0_entry3.get())
    z0 = float(z0_entry3.get())
    w0 = float(w0_entry3.get())
    n = int(np_entry3.get())
    p = int(p_entry3.get())

    x_3_sym = Symbol('x')
    y_3_sym = Function('y')(x_sym)
    dydx_3_sym = Function('dydx')(x_sym)
    dy2dx_3_sym = Function('dydx')(x_sym)

    
    equation_3 = equation_entry3.get()
    equation_3 = sympify(equation_3)

    y_v, z_v, w_v = solve_edo3(x0, x_final, y0, z0, w0, n, p)

    result_label3.config(text=f"y1 = {y_v[p]:.6f}, z1 = {z_v[p]:.6f}, w1 = {w_v[p]:.6f}",  bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))

def system_of_odes3(x, yzw):
    y, z, w = yzw
    dydx = z
    dzdx = w
    dwdx = eval(str(lambdify((x_3_sym, y_3_sym, dydx_3_sym, dy2dx_3_sym), equation_3)(x, y, dydx, dzdx)))
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

# Primeira aba
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='1° Ordem')  

# Valores pedidos ao usuario
equation_label1 = tk.Label(tab1, text="Digite a derivada de y (y'): ")
equation_label1.pack(pady=(10, 0))

equation_entry1 = tk.Entry(tab1)
equation_entry1.pack()


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
solve_button1_1 = tk.Button(tab1, text="Calcular y1", command=inciar_1nd_ordem_i, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1_1.pack(pady=(20, 0))

# Botão para calcular y1
solve_button1_2 = tk.Button(tab1, text="Listar y1", command=inciar_1nd_ordem_l, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button1_2.pack(pady=(10, 0))

# Botaão para plotar y/x
plot_button_1 = tk.Button(tab1, text="Plotar Gráfico", command=plot_graph_first_order, bd=2, bg='#107db2', fg='white',
                        font=('verdana', 8, 'bold'))
plot_button_1.pack(pady=(10, 0))


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
solve_button2_1 = tk.Button(tab2, text="Calcular y1 e z1", command=inciar_2nd_ordem_i, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2_1.pack(pady=(10, 0))

# Botão para listar y1
solve_button2_2 = tk.Button(tab2, text="Listar y1 e z1", command=inciar_2nd_ordem_l, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button2_2.pack(pady=(10, 0))

# Botão para plotar grafico 
plot_button_2 = tk.Button(tab2, text="Plotar Gráfico", command=plot_graph_second_order, bd=2, bg='#107db2',
                                      fg='white', font=('verdana', 8, 'bold'))
plot_button_2.pack(pady=(10, 0))


# Rótulo para mostrar o resultado final de y1
result_label2 = tk.Label(tab2, text="")
result_label2.pack(pady=(10, 0))


# Lista de valores
result_frame2 = tk.Frame(tab2)
result_frame2.pack()

result_text2 = tk.Text(result_frame2, wrap=tk.WORD, width=25, height=7.5)
result_text2.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Use grid

scrollbar2 = tk.Scrollbar(result_frame2, command=result_text2.yview)
scrollbar2.grid(row=0, column=1, sticky="ns")  # Use grid

result_text2.config(yscrollcommand=scrollbar2.set)

##############

# Terceira aba

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
solve_button3_1 = tk.Button(frame_aux, text="Calcular y1", command=inciar_3nd_ordem_i, bd=2, bg = '#107db2', fg ='white'
                            , font = ('verdana', 8, 'bold'))
solve_button3_1.pack(pady=(5,0))


#Botão para plotar y/x
# Botão para plotar grafico 
plot_button_3 = tk.Button(tab3, text="Plotar Gráfico", command=plot_graph_second_order, bd=2, bg='#107db2',
                                      fg='white', font=('verdana', 8, 'bold'))
plot_button_3.pack(pady=(10, 0))


# Rótulo para mostrar o resultado final de y1
result_label3 = tk.Label(frame_aux, text="")
result_label3.pack(pady=(5,0))

# Lista de valores
result_frame3 = tk.Frame(frame_aux)
result_frame3.pack(pady=(5,0))

result_text3 = tk.Text(result_frame3, wrap=tk.WORD, width=25, height=7.5)
result_text3.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Use grid

scrollbar3 = tk.Scrollbar(result_frame3, command=result_text1.yview)
scrollbar3.grid(row=0, column=1, sticky="ns")  # Use grid

result_text3.config(yscrollcommand=scrollbar3.set)

root.mainloop()


# Imprimir os resultados
#print("\nResultados:")
#for i in range(len(x_values)):
#    print(f"x = {x_values[i]:.2f}, y = {y_values[i]:.6f}, z = {z_values[i]:.6f}")
