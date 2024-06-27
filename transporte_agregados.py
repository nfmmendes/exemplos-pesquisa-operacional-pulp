import pulp as lp
import pandas as pd

## Dados do problema
df = pd.DataFrame(
    data=[
        [30, 13, 21, 433],
        [12, 40, 26, 215],
        [27, 15, 35, 782],
        [37, 25, 19, 300],
        [697, 421, 612, None],
    ],
    columns=["Deposito 1", "Deposito 2", "Deposito 3", "Oferta"],
    index=["Pedreira 1", "Pedreira 2", "Pedreira 3", "Pedreira 4", "Demanda"],
)

## Series com a oferta e a demanda.
serie_oferta = df.iloc[:-1, -1]
serie_demanda = df.iloc[-1, :-1]

### Imprime os dados para controle.
print(df)
print(serie_oferta)
print(serie_demanda)

## Cria variaveis
variables = {
    (origem, destino): lp.LpVariable(
        f"{origem}_{destino}", cat=lp.LpInteger, lowBound=0
    )
    for origem in serie_oferta.keys()
    for destino in serie_demanda.keys()
}

## Cria problema
model = lp.LpProblem("Transporte agregados")

## Funcao objetivo
model += lp.lpSum(
    variables[(origem, destino)] * df[destino][origem]
    for origem in serie_oferta.keys()
    for destino in serie_demanda.keys()
)

### Restricao de demanda
for destino, valor in serie_demanda.items():
    model += lp.lpSum(variables[(origem, destino)] for origem in serie_oferta.keys()) == valor

### Restricao de oferta
for origem, valor in serie_oferta.items():
    model += lp.lpSum(variables[(origem, destino)] for destino in serie_demanda.keys()) <= valor

### Resolve o problema
print(model)

model.solve()

print("Status: ", lp.LpStatus[model.status])

for v in model.variables():
    if v.varValue > 0:
        print(v.name, " = ", v.varValue)

