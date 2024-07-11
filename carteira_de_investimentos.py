import pulp as lp
import pandas as pd

df = pd.DataFrame(
    data=[
        [0.0, 100, 0.0, False],
        [4.0, 99.5, 0.5, False],
        [4.5, 96.0, 4.0, False],
        [5.5, 90, 5.0, False],
        [7.0, 85.0, 7.5, False],
        [10.5, 0.0, 10.0, True],
        [8.5, 0.0, 10.0, True],
        [9.2, 0.0, 10.0, True],
    ],
    columns=["Retorno(%)", "Parte liquida(%)", "Capital minimo(%)", "Risco(%)"],
    index=[
        "Caixa",
        "Curto prazo",
        "TD 1-5 anos",
        "TD 5-10 anos",
        "TD mais de 10 anos",
        "Emprestimos pessoais",
        "Financiamento imoveis",
        "Emprestimos comerciais",
    ],
)

print(df)

valores_disponiveis = pd.Series(
    {"Conta corrente": 150, "Capital proprio": 20, "Fundos de investimento": 80}
)
valor_investido = {
    # O lower bound é definido considerando o investimento minimo
    # em cada opcao.
    tipo: lp.LpVariable(name=f"x_{tipo}", lowBound=0.05 * valores_disponiveis.sum())
    for tipo in df.index
}

## Cria uma instancia do problema
model = lp.LpProblem(sense=lp.LpMaximize)

## Funcao objetivo
## No livro este problema é multi-objetivo, mas aqui foi utilizado apenas o primeiro objetivo.
model += lp.lpSum(
    valor_investido[opcao] * df["Retorno(%)"][opcao] / 100 for opcao in df.index
)

## Valor maximo investido
model += lp.lpSum(valor_investido) <= valores_disponiveis

## Valor minimo em caixa:
model += (
    valor_investido["Caixa"]
    >= 0.14 * valores_disponiveis["Conta corrente"]
    + 0.04 * valores_disponiveis["Fundos de investimento"]
)

## Minimo de investimento liquido
model += (
    lp.lpSum(
        valor_investido[opcao] * df["Parte liquida(%)"][opcao] / 100
        for opcao in df[(df["Parte liquida(%)"] > 0)].index
    )
    >= 0.47 * valores_disponiveis["Conta corrente"]
    + 0.36 * valores_disponiveis["Fundos de investimento"]
)

## Valor minimo de emprestimos comerciais
model += valor_investido["Emprestimos comerciais"] >= 0.30 * valores_disponiveis.sum()

print(model)

## Resolve o problema
model.solve()

## Escreve o resultado
print("Status: ", lp.LpStatus[model.status])

for v in model.variables():
    print(v.name, " = ", v.varValue)
