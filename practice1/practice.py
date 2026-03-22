import pandas as pd

# первое задание
# df = pd.read_csv("articles/tables/table-1.2.1.csv", sep=",")

# print(df)


# второе задание
# with open("result_set.csv", "w", encoding="UTF-8") as f:
#     for i in range(6):
#         if i == 0:
#             with open("table_export.csv", "r", encoding="UTF-8") as r:
#                 f.writelines(r.readlines())
#         else:
#             with open(f"table_export ({i}).csv", "r", encoding="UTF-8") as r:
#                 f.writelines(r.readlines()[1:-1])

df = pd.read_csv("result_set.csv", sep=",")
df = df[[
    "Formula",
    "Space Group Number",
    "Formation Energy",
    "Volume",
    "Density",
    "Total Magnetization"
]
]
df.to_csv('finish_dataset.csv', index=False)
print(df.info())
print(df.head())
