import sympy as sp

# Definir o símbolo para x
x_sym = sp.Symbol('x')

# Definir a função y(x) como um símbolo
y_sym = sp.Function('y')(x_sym)

# Definir a função dy/dx como um símbolo
dydx_sym = sp.Function('dydx')(x_sym)

# Utilizar os símbolos em expressões
expr1 = y_sym + dydx_sym
expr2 = y_sym * dydx_sym

# Imprimir as expressões
print(expr1)
print(expr2)
