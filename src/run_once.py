import pandas as pd
from src.optimiser import solve_weights

df = pd.read_csv("data/processed/universe.csv")

w, info = solve_weights(df, month=0)

out = df[["Equity", "Ticker", "Industry"]].copy()
out["weight"] = w
out = out.sort_values("weight", ascending=False)

print(info)
print(out.head(len(df)))
