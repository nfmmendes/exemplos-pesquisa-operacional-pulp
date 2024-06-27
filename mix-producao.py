import pulp as lp
import pandas as pd


df = pd.DataFrame(
    data=[[100, 50], [10, 8], [1, 1], [1500, 6000]],
    columns=["luxo", "basico"],
    index=["Custo", "Uso mao de obra", "Uso maquinas", "Demanda"],
)

recursos = pd.DataFrame([[25000, 4500]], columns=df.index[1:3])

x = {
    tipo: lp.LpVariable(tipo, cat=lp.LpInteger, lowBound=0, upBound=df[tipo]["Demanda"])
    for tipo in df.columns
}

model = lp.LpProblem(sense=lp.LpMaximize)

model += lp.lpSum(df[tipo]["Custo"] * x[tipo] for tipo in df.columns)

for recurso, valor in recursos.items():
    model += lp.lpSum(df[tipo][recurso] * x[tipo] for tipo in df.columns) <= valor

print(model)

model.solve()

print("Status: ", lp.LpStatus[model.status])

for v in model.variables():
    if v.varValue > 0:
        print(v.name, " = ", v.varValue)
