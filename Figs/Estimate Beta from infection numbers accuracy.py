import pandas as pd

df = pd.read_csv("Estimating Beta from infection numbers.csv")

def row_func(row):
    row['Method 1 Error'] = abs(row['Actual Beta'] - row['Method 1 Beta'])
    row['Method 2 Error'] = abs(row['Actual Beta'] - row['Method 2 Beta'])
    return row

print(df.apply(row_func, axis=1).mean())