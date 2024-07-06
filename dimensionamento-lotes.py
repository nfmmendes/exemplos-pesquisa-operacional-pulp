import pulp as lp
import pandas as pd

# Items a serem produzidos.
items = ["Item 1", "Item 2"]
periodos = ["Periodo 1", "Periodo 2", "Periodo 3"]
tempo_disponivel_por_periodo = 40 * 60  # 40 horas em minutos.

# Dados da
df = pd.DataFrame(
    data=[
        [  # Demandas
            pd.DataFrame(
                data=[[100, 90, 120], [40, 50, 80]],
                columns=periodos,
                index=items,
            )
        ],
        [  # Custos de producao.
            pd.DataFrame(
                data=[[20, 20, 30], [20, 20, 30]],
                columns=periodos,
                index=items,
            )
        ],
        [  # Custos de estoque (nao precisa especificar o custo do ultimo periodo).
            pd.DataFrame(
                data=[[2, 3], [2.5, 3.5]],
                columns=periodos[:-1],
                index=items,
            )
        ],
        [pd.DataFrame(data=[[12], [15]], columns=["Tempo em minutos"], index=items)],
    ],
    index=["Demandas", "Custos de producao", "Custos de estoque", "Tempos de producao"],
)[0]

# Os tempos de producao neste exemplo foram ajustados pois os originai (15 e 20 minutos), levavam 
# a um tempo de producao minimo de 2300 minutos no periodo 1, um tempo de producao de 2350 
# minutos no periodo 2 e um 3400 minutos no periodo 3. Mesmo com producao maxima nos periodos 1 e 2
# nao era possivel satisfazer a demanda do periodo 3, o que deixava o problema modelo infactivel. 

print("Demandas: \n", df["Demandas"])
print("Custos de producao: \n", df["Custos de producao"])
print("Custos de estoque: \n", df["Custos de estoque"])

variaveis_de_producao = {
    (item, periodo): lp.LpVariable(f"x_{item}_{periodo}", lowBound=0.0, cat=lp.LpInteger)
    for periodo in periodos
    for item in items
}

variaveis_de_estoque = {
    (item, periodo): lp.LpVariable(f"y_{item}_{periodo}", lowBound=0.0, cat=lp.LpInteger)
    for periodo in periodos[:-1]
    for item in items
}

model = lp.LpProblem()

model += lp.lpSum(
    variaveis_de_producao[(item, periodo)] * df["Custos de producao"][periodo][item]
    for periodo in periodos
    for item in items
) + lp.lpSum(
    variaveis_de_estoque[(item, periodo)] * df["Custos de estoque"][periodo][item]
    for periodo in periodos[:-1]
    for item in items
)

# Restricao de capacidade de producao.
tempos_de_producao = df["Tempos de producao"]["Tempo em minutos"]
for periodo in periodos:
    model += lp.lpSum( variaveis_de_producao[(item, periodo)] * tempos_de_producao[item] for item in items) \
          <= tempo_disponivel_por_periodo

# Restricao de fluxo.
for i, periodo in enumerate(periodos):
    for item in items:
        model += variaveis_de_producao[(item, periodo)] \
            + (variaveis_de_estoque[(item, periodos[i - 1])] if i > 0 else 0) \
            - (variaveis_de_estoque[(item, periodo)] if i < len(periodos) - 1 else 0) \
            == df["Demandas"][periodo][item]

# No livro o modelo é descrito com uma demanda de 70 items 2 no periodo 2, mesmo com a tabela
# indicando 50. 

print(model)

model.solve()


print("Status = ", lp.LpStatus[model.status])

for v in model.variables():
    if v.varValue > 0:
        print(f"{v.name} = {v.varValue}")
