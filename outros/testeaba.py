import tkinter as tk
from tkinter import ttk

# Função para exibir uma mensagem quando a primeira aba é selecionada
def tab1_function():
    message_label.config(text="Aba 1 selecionada")

# Função para exibir uma mensagem quando a segunda aba é selecionada
def tab2_function():
    message_label.config(text="Aba 2 selecionada")

# Criar a janela principal
root = tk.Tk()
root.title("Abas com Tkinter")

# Criar um Notebook (aba)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Primeira aba
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='Aba 1')

tab1_button = tk.Button(tab1, text="Selecionar Aba 1", command=tab1_function)
tab1_button.pack()
 
# Segunda aba
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='Aba 2')

tab2_button = tk.Button(tab2, text="Selecionar Aba 2", command=tab2_function)
tab2_button.pack()

# Rótulo para mostrar a mensagem da aba selecionada
message_label = tk.Label(root, text="")
message_label.pack()

# Iniciar o loop da interface gráfica
root.mainloop()
