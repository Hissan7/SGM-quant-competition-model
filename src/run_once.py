import pandas as pd
from src.optimiser import solve_weights

df = pd.read_csv("data/processed/universe.csv")

"""To show weights for all months (entire 9 month period)"""
# for i in range(1,10):
#     for j in range(0,2):
#         print()
#     print(f"---------------MONTH {i}---------------")
#     w, info = solve_weights(df, month=i)

#     out = df[["Equity", "Ticker", "Industry"]].copy()
#     out["weight"] = w
#     out = out.sort_values("weight", ascending=True)

#     print(info)
#     print(out.head(len(df)))
    
w, info = solve_weights(df, month=i)

out = df[["Equity", "Ticker", "Industry"]].copy()
out["weight"] = w
out = out.sort_values("weight", ascending=True)

print(info)
print(out.head(len(df)))