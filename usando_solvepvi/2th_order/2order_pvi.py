import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

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
        x_values, y_values, z_values = solve_pvi_rk4th_edo_2th_order(f, x0, y0, z0, x_final, n)
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

def solve_pvi_rk4th_edo_2th_order(f, x0, y0, z0, x_max, n):
    
    # Use solve_ivp to solve the IVP
    sol = solve_ivp(lambda t, u: [u[1], f(t, u[0], u[1])], (x0, x_max), [y0, z0], t_eval=np.linspace(x0, x_max, n + 1), method='RK45')

    return sol.t, sol.y[0], sol.y[1]

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


root.mainloop()
