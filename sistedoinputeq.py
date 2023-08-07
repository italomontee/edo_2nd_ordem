import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
import tkinter as tk
# Função que representa o sistema de EDOs
def system_of_odes(x, y_z):
    y, z = y_z
    dydx = z
    dzdx = eval(str(sp.lambdify((x_sym, y_sym, dydx_sym), equation)(x, y, dydx)))
    return [dydx, dzdx]


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

x_sym = sp.Symbol('x')
y_sym = sp.Function('y')(x_sym)
dydx_sym = sp.Function('dydx')(x_sym)

#Eq 2nd ordem
equation = input("Entre com a equação: ")
equation = sp.sympify(equation)

# Solicitar informações do usuário
x0 = float(x0_inicial_entry.get())
x_final = float(x0_final_entry.get())
y0 = float(y0_entry.get())
z0 = float(z0_entry.get())



# Definir o intervalo de integração
x_span = (x0, x_final)

# Definir as condições iniciais
initial_conditions = [y0, z0]

# Resolver o sistema usando o método de Runge-Kutta de 4ª ordem
solution = solve_ivp(system_of_odes, x_span, initial_conditions, method='RK45')

# Obter os resultados
x_values = solution.t
y_values, z_values = solution.y

# Imprimir os resultados
print("\nResultados:")
for i in range(len(x_values)):
    print(f"x = {x_values[i]:.2f}, y = {y_values[i]:.6f}, z = {z_values[i]:.6f}")

