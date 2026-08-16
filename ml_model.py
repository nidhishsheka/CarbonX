import sqlite3
import pandas as pd

DB_NAME = "cc.db"

conn = sqlite3.connect(DB_NAME)

query = """
SELECT
    DATE(created_at) AS trade_date,
    AVG(price_per_credit) AS avg_price,
    SUM(credit_amount) AS credits_traded,
    SUM(total_amount) AS trade_value
FROM Credit_Transaction
WHERE status = 'Completed'
GROUP BY DATE(created_at)
ORDER BY trade_date ASC
"""

df = pd.read_sql_query(query, conn)

conn.close()

print("\nMarket Dataset:")
print(df)

if len(df) < 5:
    print("\nNot enough historical data for ML training.")
    print("At least 5 trading periods are recommended.")
else:
    df["previous_price"] = df["avg_price"].shift(1)
    df["moving_average"] = df["avg_price"].rolling(3).mean()
    df["price_change"] = df["avg_price"].pct_change() * 100

    df["target_price"] = df["avg_price"].shift(-1)

    df = df.dropna()

    print("\nML Training Dataset:")
    print(df)