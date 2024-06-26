import pulp as pl
import pandas as pd


df = pd.DataFrame(
    data=[[0.2, 0.6, 0.56], [0.5, 0.4, 0.81], [0.4, 0.4, 0.46]],
    columns=["PROTEINA", "CALCIO", "CUSTO"],
    index=["OSSO", "SOJA", "PEIXE"],
)

exigencias_minimas = {"PROTEINA": 0.3, "CALCIO": 0.5}

## Mostra os dados para garantir que estejam corretos
print(df)
print(exigencias_minimas)

## Cria o modelo
model = pl.LpProblem("dieta")

## Insere as quantidades
quantidades = [pl.LpVariable(ingrediente, lowBound=0.0) for ingrediente in df.index]

## Funcao objetivo
model += pl.lpSum(
    quantidade * df["CUSTO"][quantidade.name] for quantidade in quantidades
)

## Restricoes
for nutriente, exigencia in exigencias_minimas.items():
    model += pl.lpSum(quantidade * df[nutriente][quantidade.name] for quantidade in quantidades) >= exigencia

model.solve()

print("Status", pl.LpStatus[model.status])

for v in model.variables():
    print(v.name, "=", v.varValue)
