import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import tkinter as tk
from tkinter import tkk

# Função que representa o sistema de EDOs
def system_of_odes(x, y_z):
    y, z = y_z
    dydx = z
    dzdx = eval(str(sp.lambdify((x_sym, y_sym, dydx_sym), equation)(x, y, dydx)))
    return [dydx, dzdx]


#Função para iniciar o processo
def iniciar():
    global x_sym, y_sym, dydx_sym, equation

    # Solicitar informações do usuário
    p = int(p_entry.get())
    x0 = float(x0_inicial_entry.get())
    x_final = float(x0_final_entry.get())
    y0 = float(y0_entry.get())
    z0 = float(z0_entry.get())
    n = int(n_entry.get())

    #Definir simbolos
    x_sym = sp.Symbol('x')
    y_sym = sp.Function('y')(x_sym)
    dydx_sym = sp.Function('dydx')(x_sym)

    equation = equation_entry.get()
    equation = sp.sympify(equation)

    solve_edo(x0, x_final, y0, z0, n, p)


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
    solution = solve_ivp(system_of_odes, x_span, initial_conditions, method='RK45', t_eval=np.linspace(x0, x_final, n))

    # Obter os resultados
    x_values = solution.t
    y_values, z_values = solution.y

    result_label.config(text=f"y1 = {y_values[p]:.6f}, z1 = {z_values[p]}")


##Parte Grafica

root = tk.Tk()
root.title("Resolver EDO's de 1nd e 2nd Ordem")

equation_label = tk.Label(root, text="Entre com a EDO (use x, y e dydx)")
equation_label.pack()

equation_entry = tk.Entry(root)
equation_entry.pack()


x0_inicial_label = tk.Label(root, text="Digite o valor inicial de x, (x0): ")
x0_inicial_label.pack()

x0_inicial_entry = tk.Entry(root)
x0_inicial_entry.pack()

x0_final_label = tk.Label(root, text="Digite o valor final de x, (xf): ")
x0_final_label.pack()

x0_final_entry = tk.Entry(root)
x0_final_entry.pack()

y0_label = tk.Label(root, text="Digite o valor inicial de y, y(x0): ")
y0_label.pack()

y0_entry = tk.Entry(root)
y0_entry.pack()

z0_label = tk.Label(root, text="Digite o valor inicial de z, y'(x0): ")
z0_label.pack()

z0_entry = tk.Entry(root)
z0_entry.pack()

n_label = tk.Label(root, text="numero de pontos (n): ")
n_label.pack()

n_entry = tk.Entry(root)
n_entry.pack()


p_label = tk.Label(root, text="ponto desejado (p): ")
p_label.pack()

p_entry = tk.Entry(root)
p_entry.pack()

# Botão para calcular y1
solve_button = tk.Button(root, text="Calcular y1", command=iniciar)
solve_button.pack()

# Rótulo para mostrar o resultado final de y1
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()


# Imprimir os resultados
#print("\nResultados:")
#for i in range(len(x_values)):
#    print(f"x = {x_values[i]:.2f}, y = {y_values[i]:.6f}, z = {z_values[i]:.6f}")
