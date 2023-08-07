import tkinter as tk
from tkinter import messagebox
import sympy as sp

def reduce_to_first_degree():
    equation = entry_equation.get()


    # Definindo a variável 'y' como função de 'x'
    x = sp.Symbol('x')
    y = sp.Function('y')(x)

    # Introduzindo uma nova variável auxiliar v(x) para representar a derivada de 'y'
    v = sp.Function('v')(x)

    # Convertendo a equação para forma simbólica


    # Substituindo 'y' por 'v' e 'y'' por 'v'
    eq = eq.subs(y, v).subs(y.diff(x), y)

    result_label.config(text=f"A EDO de 1ª ordem é: {eq}")

# Criação da janela tkinter
root = tk.Tk()
root.title("Redução de EDO de 2ª ordem para 1ª ordem")

# Criando widgets
label_equation = tk.Label(root, text="Insira a EDO de 2ª ordem (use 'y' como variável dependente e 'x' como variável independente):")
label_equation.pack()

entry_equation = tk.Entry(root)
entry_equation.pack()

calculate_button = tk.Button(root, text="Calcular EDO de 1ª ordem", command=reduce_to_first_degree)
calculate_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
