import sympy as sp

# Definindo as variáveis simbólicas
x = sp.Symbol('x')
y = sp.Function('y')(x)

# Expressão a ser integrada
expressao = y.diff(x).diff(x).diff(x).diff(x) + x**2

# Realizando a integração
resultado = sp.integrate(expressao, x)

# Imprimindo o resultado
print("Resultado da integração:", resultado)
