import pulp as lp
import pandas as pd

## Pontos observados
df = pd.DataFrame(data=[[0, 1, 2, 3], [-1, 0, 1, 1]], index=["x", "y"])

## Mostra o dataframe 
print(df)

## Variaveis para os coeficientes das retas.
a = lp.LpVariable(name="a")
b = lp.LpVariable(name="b")

## Variaveis (positivas) para os erros.
variaveis_erro = []
for i in range(0, len(df.columns)):
    variaveis_erro.append(
        {
            "+": lp.LpVariable(name=f"e_pos_{i}", lowBound=0.0),
            "-": lp.LpVariable(name=f"e_neg_{i}", lowBound=0.0),
        }
    )

## Cria um modelo
model = lp.LpProblem()

## Funcao objetivo
model += lp.lpSum(
    variaveis_erro[i][sinal] for i in range(0, len(df.columns)) for sinal in ["+", "-"]
)

## Restricoes
for i in range(0, len(df.columns)):
    model += (
        a * df[i]["x"] + b + variaveis_erro[i]["+"] - variaveis_erro[i]["-"]
        == df[i]["y"]
    )

## Imprime o modelo
print(model)

## Resolve o problema
model.solve()

print("Status: ", lp.LpStatus[model.status])

# Imprime a solucao
for v in model.variables():
    print(v.name, " = ", v.varValue)
