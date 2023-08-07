import tkinter as tk
from sympy import *
import numpy as np

def convert_to_first_order():
    equation_str = derivative_entry.get()
    x = Symbol('x')
    y = Function('y')(x)

    # Convertendo a string da equação para a forma simbólica
    equation_str = equation_str.replace("y''", "y'")
    equation_str = equation_str.replace("y'", "y")
    eq1 = eval(equation_str)

    solve_runge_kutta(equation_str)
    


def solve_runge_kutta(equation_str):
    # Obter os valores informados pelo usuário
    x0 = float(x0_entry.get())
    y0 = float(y0_entry.get())
    h = float(h_entry.get())
    n = int(n_entry.get() )
    derivative_str = equation_str

    # Definir a variável simbólica para x e y
    x, y = symbols('x y')

    # Definir a função f(x, y) como a derivada de y (y')
    derivative_expr = sympify(derivative_str)
    f = lambdify((x, y), derivative_expr, modules=['numpy'])

    # Resolvendo a equação usando o método de Runge-Kutta
    _, y_values, num_points = runge_kutta_4th_order(f, x0, y0, h, x0, n)

    # O valor de y1 será o último elemento do array y_values
    y1 = y_values[-1]

    # Atualizar o rótulo com o valor de y1
    result_label.config(text=f"y1 = {y1:.6f}")
    points_label.config(text=f"np = {num_points}")

def runge_kutta_4th_order(f, x0, y0, h, x_min, n):
    # Implementar o método de Runge-Kutta de quarta ordem aqui
    # Retornar os valores de x e y para o intervalo de interesse
    
    x_values = np.linspace(x_min, x_min+h*n,n+1)
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

    return x_values, y_values, n

# Criar a janela principal
root = tk.Tk()
root.title("Resolução de Equações Diferenciais com Runge-Kutta")

# Criar elementos da interface gráfica
x0_label = tk.Label(root, text="Digite o valor inicial de x (x0): ")
x0_label.pack()

x0_entry = tk.Entry(root)
x0_entry.pack()

y0_label = tk.Label(root, text="Digite o valor inicial de y (y0): ")
y0_label.pack()

y0_entry = tk.Entry(root)
y0_entry.pack()

h_label = tk.Label(root, text="Digite o passo de integração (h): ")
h_label.pack()

h_entry = tk.Entry(root)
h_entry.pack()

n_label = tk.Label(root, text="Digite a quantidade de iterações (n): ")
n_label.pack()

n_entry = tk.Entry(root)
n_entry.pack()

derivative_label = tk.Label(root, text="Digite a derivada de y (y'): ")
derivative_label.pack()

derivative_entry = tk.Entry(root)
derivative_entry.pack()

# Botão para calcular y1
solve_button = tk.Button(root, text="Calcular y1", command=convert_to_first_order)
solve_button.pack()

# Rótulo para mostrar o resultado final de y1
points_label = tk.Label(root, text="")
points_label.pack()

# Rótulo para mostrar o resultado final de y1
result_label = tk.Label(root, text="")
result_label.pack()

# Iniciar o loop da interface gráfica
root.mainloop()
