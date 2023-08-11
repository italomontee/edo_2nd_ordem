import numpy as np
from scipy.integrate import solve_ivp
from sympy import *
import tkinter as tk
from tkinter import ttk

def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

# Criar a janela principal
root = tk.Tk()
root.title("Resolver EDO's de 1nd e 2nd Ordem")

root.geometry("300x575")

# Criar um notebok 
main = ttk.Notebook(root)
main.pack(fill='both', expand=1)

# Criar aba
tab1 = ttk.Frame(main)
main.add(tab1, text='aba')

# Criar aba
tab2 = ttk.Frame(main)
main.add(tab2, text='aba')

# crinado canvas
canvas= tk.Canvas(tab1)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scroll = ttk.Scrollbar(tab1, orient=tk.VERTICAL, command=canvas.yview)
scroll.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scroll.set )
canvas.bind('<Configure>',on_configure)

second = tk.Frame(canvas)

canvas.create_window((0,0), window=second, anchor = "nw" )

for thing in range(100):
    tk.Button(second, text=f'Button {thing} ').grid(row=thing, column=5, pady = 10, padx = 10)
    tk.Label(second, text=f'ola {thing}').grid(row=thing, column=0, pady = 10, padx = 10)
root.mainloop()
