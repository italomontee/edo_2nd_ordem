import tkinter as tk
from tkinter import ttk

def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

root = tk.Tk()
root.title("Scrollable Frame Example")

root.geometry("300x400")

main = ttk.Frame(root)
main.pack(fill='both', expand=1)

canvas = tk.Canvas(main)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scroll = ttk.Scrollbar(main, orient=tk.VERTICAL, command=canvas.yview)
scroll.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scroll.set)
canvas.bind('<Configure>', on_configure)

second = tk.Frame(canvas)
canvas.create_window((0, 0), window=second, anchor="nw")

for thing in range(100):
    tk.Button(second, text=f'Button {thing}').grid(row=thing, column=0, pady=2, padx=2)

second.update_idletasks()  # Update the frame to calculate its size

canvas.config(scrollregion=canvas.bbox("all"))  # Update the scrollable region

root.mainloop()
