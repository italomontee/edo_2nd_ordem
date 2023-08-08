import sympy as sp
def convert_to_first_order(equation_str):

    x = sp.Symbol('x')
    y = sp.Function('y')(x)

    # Convertendo a string da equação para a forma simbólica
    equation_str = equation_str.replace("y''", "y.diff(x).diff(x)")
    equation_str = equation_str.replace("y'", "y")
    eq1 = eval(equation_str)

    

    return equation_str

# Solicitar a função do usuário
equation_str = input("Insira a EDO de segunda ordem (use 'y' como variável dependente e 'x' como variável independente): ")

# Executando a conversão e imprimindo o resultado
result_equation = convert_to_first_order(equation_str)
print("A EDO de primeira ordem equivalente é:", result_equation)
