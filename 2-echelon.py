import pulp as lp
import pandas as pd

## Dataframe com dados da rede produtor --> deposito
produtorDeposito_df = pd.DataFrame(
    data=[[1, 3, 800], [1, 2, 1000]],
    columns=["Campinas", "Barra Mansa", "Oferta"],
    index=["Araraquara", "SJ Campos"],
)

## Dataframe com os dados da rede depositor --> consumidor
depositoConsumidor_df = pd.DataFrame(
    data=[[1, 3, 3], [3, 4, 1]],
    columns=["Sao Paulo", "Belo Horizonte", "Rio de Janeiro"],
    index=produtorDeposito_df.columns[:-1],
)

## Series com as ofertas e demandas
serie_oferta = produtorDeposito_df.iloc[:, -1]
serie_demandas = pd.DataFrame(
    data=[[500, 400, 900]], columns=depositoConsumidor_df.columns
)

## Imprime o codigo para conferencia
print(produtorDeposito_df)
print(depositoConsumidor_df)

## Variaveis do primeiro "echelon"
variaveisPrimeiraFase = {
    (origem, destino): lp.LpVariable(
        f"{origem}_{destino}", cat=lp.LpInteger, lowBound=0.0
    )
    for origem in produtorDeposito_df.index
    for destino in produtorDeposito_df.columns[:-1]
}

## Variaveis do segundo "echelon"
variaveisSegundaFase = {
    (origem, destino): lp.LpVariable(
        f"{origem}_{destino}", cat=lp.LpInteger, lowBound=0.0
    )
    for origem in depositoConsumidor_df.index
    for destino in depositoConsumidor_df.columns
}

## Criacao do modelo
model = lp.LpProblem()


## Funcao objetivo
model += lp.lpSum(
    produtorDeposito_df[destino][origem] * variaveisPrimeiraFase[(origem, destino)]
    for origem in produtorDeposito_df.index
    for destino in produtorDeposito_df.columns[:-1]
) + lp.lpSum(
    depositoConsumidor_df[destino][origem] * variaveisSegundaFase[(origem, destino)]
    for origem in depositoConsumidor_df.index
    for destino in depositoConsumidor_df.columns
)

## Restricao de oferta
for origem in produtorDeposito_df.index:
    model += (
        lp.lpSum(
            variaveisPrimeiraFase[(origem, destino)]
            for destino in produtorDeposito_df.columns[:-1]
        )
        <= serie_oferta[origem]
    )

## Restricao de demanda
for destino in depositoConsumidor_df.columns:
    model += (
        lp.lpSum(
            variaveisSegundaFase[(origem, destino)]
            for origem in depositoConsumidor_df.index
        )
        == serie_demandas[destino]
    )

## Restricao de conservacao de fluxo
for deposito in depositoConsumidor_df.index:
    model += lp.lpSum(
        variaveisPrimeiraFase[((origem, deposito))]
        for origem in produtorDeposito_df.index
    ) == lp.lpSum(
        variaveisSegundaFase[((deposito, destino))]
        for destino in depositoConsumidor_df.columns
    )

print(model)

model.solve()

print("Status: ", lp.LpStatus[model.status])

for v in model.variables():
    if v.varValue > 0:
        print(v.name, " = ", v.varValue)
