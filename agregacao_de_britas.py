import pulp as lp
import pandas as pd

## Data frame com proporcoes e custos
df = pd.DataFrame(
    data=[[0.10, 0.20, 0.70, 6], [0.35, 0.60, 0, 7], [0.78, 0.02, 0.0, 18]],
    index=["GRANITICA", "SEIXO", "COMERCIAL"],
    columns=["19-38mm", "38-76mm", "76-152mm", "CUSTO"],
)

## Proporcao minima de cada granularidade
proporcaoMinima = {"19-38mm": 0.2, "38-76mm": 0.35, "76-152mm": 0.35}

## Coeficiente adiciona de proporcao entre os agregados
proporcaoAdicional = {"GRANITICA": 0.0, "SEIXO": 0.20, "COMERCIAL": 0.30}

## Imprime data frame
print(df)

## Cria variaveis
proporcoes = [
    lp.LpVariable(tipoBrita, lowBound=0.0, upBound=1.0) for tipoBrita in df.index
]

## Cria modelo
model = lp.LpProblem("Agregacao_de_britas")

## Funcao objetivo
model += lp.lpSum(proporcao * df["CUSTO"][proporcao.name] for proporcao in proporcoes)

## Restricoes
for tipo in df.columns[:-1]:
    model += lp.lpSum(v * df[tipo][v.name] for v in proporcoes) >= proporcaoMinima[tipo]

model += lp.lpSum(v * proporcaoAdicional[v.name] for v in proporcoes) >= 0.10

print(model)

model.solve()

print("Status", lp.LpStatus[model.status])

for v in model.variables():
    print(v.name, " = ", v.varValue)
