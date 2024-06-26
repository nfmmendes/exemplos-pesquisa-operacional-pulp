import pulp as lp
import pandas as pd

## Data frame com os dados do problema
df = pd.DataFrame(
    data=[[4, 2, 5, 800], [11, 7, 4, 1000], [500, 400, 900, 0]],
    columns=["Sao Paulo", "Belo Horizonte", "Rio de Janeiro", "Disponibilidade"],
    index=["Araraquara", "SJ Campos", "Demanda"],
)

## Fatias do data frame por tipo de dados
custos_df = df.iloc[:-1, :-1]
demandas_df = df.iloc[-1, :-1]
estoque_df = df.iloc[:-1, -1]
origens = df.index.values[:-1]
destinos = df.columns.values[:-1]

## Imprime dados para conferencia
print(custos_df)
print(demandas_df)
print(estoque_df)

model = lp.LpProblem()

variaveis = {
    (origem, destino): lp.LpVariable(
        f"{origem}_{destino}", lowBound=0.0, cat=lp.LpInteger
    )
    for origem in origens
    for destino in destinos
}

## Funcao objetivo
model += lp.lpSum(
    variaveis[(origem, destino)] * custos_df[destino][origem]
    for origem in origens
    for destino in destinos
)

### Restricao de demanda
for destino, valor in demandas_df.items():
    model += lp.lpSum(variaveis[(origem, destino)] for origem in origens) == valor

### Restricao de estoque
for origem, valor in estoque_df.items():
    model += lp.lpSum(variaveis[(origem, destino)] for destino in destinos) <= valor

print(model)

model.solve()

print("Status", lp.LpStatus[model.status])

for v in model.variables():
    print(v.name, " = ", v.varValue)
