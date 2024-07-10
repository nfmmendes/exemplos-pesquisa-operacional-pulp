import pulp as lp
import pandas as pd

df = pd.DataFrame(data=[[0, 1, 2, 3], [-1, 0, 1, 1]], index=["x", "y"])

print(df)

a = lp.LpVariable(name="a")
b = lp.LpVariable(name="b")

variaveis_erro = []
for i in range(0, len(df.columns)):
    variaveis_erro.append(
        {
            "+": lp.LpVariable(name=f"e_pos_{i}", lowBound=0.0),
            "-": lp.LpVariable(name=f"e_neg_{i}", lowBound=0.0),
        }
    )

model = lp.LpProblem()

model += lp.lpSum(
    variaveis_erro[i][sinal] for i in range(0, len(df.columns)) for sinal in ["+", "-"]
)

for i in range(0, len(df.columns)):
    model += (
        a * df[i]["x"] + b + variaveis_erro[i]["+"] - variaveis_erro[i]["-"]
        == df[i]["y"]
    )

print(model)

model.solve()

print("Status: ", lp.LpStatus[model.status])

for v in model.variables():
    print(v.name, " = ", v.varValue)
