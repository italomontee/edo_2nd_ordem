import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

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
        x_values, y_values, z_values, w_values, j_values = solve_pvi_rk4th_edo_4rd_order(f, x0, y0, z0, w0, j0, x_final, n)
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

def solve_pvi_rk4th_edo_4rd_order(f, x0, y0, z0, w0, j0, x_max, n):
    sol = solve_ivp(lambda t, u: [u[1], u[2], u[3], f(t, u[0], u[1], u[2], u[3])],
                    (x0, x_max), [y0, z0, w0, j0 ],
                    t_eval=np.linspace(x0, x_max, n + 1),
                    method='RK45')
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]

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

g = 0

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

root.mainloop()
