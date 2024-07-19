import pulp as lp
import pandas as pd

"""
 Dados do problema
 No livro o problema aparece apenas com a descricao algebrica, sem valores
 numericos
"""
df = pd.DataFrame(
    data=[[3, 7, 6, 5], [2, 6, 1, 3], [5, 3, 4]],
    columns=["Tratamento 1", "Tratamento 2", "Tratamento 3", "Tolerancia"],
    index=["A", "B", "custo"],
)

demanda = 1500
vazao = 2400

print(df)

## Variaveis
consumo_por_tratamento = [
    lp.LpVariable(name=f"x_{tratamento}", lowBound=0.0)
    for tratamento in df.columns[:-1]
]

## Inicia o modelo
model = lp.LpProblem()

## Funcao objetivo
model += lp.lpSum(
    df.iloc[-1, i] * consumo_por_tratamento[i] for i in range(0, len(df.columns) - 1)
)

## Restricoes de demanda e disponibilidade
model += lp.lpSum(consumo_por_tratamento[:]) >= demanda
model += lp.lpSum(consumo_por_tratamento[:]) <= vazao

"""
O modelo no livro considera a vazao maxima para calcular a quantidade final de poluentes,
por volume mas deveria considerar o consumo, caso contrario podera aceitar niveis de
poluentes maiores que o aceitavel.
"""
model += lp.lpSum(
    consumo_por_tratamento[t] * df.iloc[0, t] for t in range(0, len(df.columns) - 1)
) <= df["Tolerancia"]["A"] * lp.lpSum(consumo_por_tratamento[:])

model += lp.lpSum(
    consumo_por_tratamento[t] * df.iloc[1, t] for t in range(0, len(df.columns) - 1)
) <= df["Tolerancia"]["B"] * lp.lpSum(consumo_por_tratamento[:])

print(model)

model.solve()

print("Status: ", lp.LpStatus[model.status])

for v in model.variables():
    if v.varValue > 0:
        print(v.name, " = ", v.varValue)
