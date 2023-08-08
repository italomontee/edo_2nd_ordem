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


#definir simbolos
x_sym = sp.Symbol('x')
y_sym = sp.Function('y')(x_sym)
dydx_sym = sp.Function('dydx')(x_sym)

# Solicitar informações do usuário
equation = input("eq: ")
equation = sp.sympify(equation)

x0 = float(input("x0: "))
x_final = float(input("xf: "))
y0 = float(input("y0: "))
z0 = float(input("z0: "))


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
for i in range(0, len(x_values)):
    print(f"x = {x_values[i]:.2f}, y = {y_values[i]:.6f}, z = {z_values[i]:.6f}")

