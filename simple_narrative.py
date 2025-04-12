import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules

# --- Step 1: Load Data ---
engine = create_engine('sqlite:///bybit_data.db')
df = pd.read_sql("SELECT * FROM btcusdt_daily_1year", engine)

# Convert datetime column and create day-of-week column
df['datetime'] = pd.to_datetime(df['datetime'])
df['trading_day'] = df['datetime'].dt.date
df['day_of_week'] = df['datetime'].dt.day_name()

# --- Step 2: Convert SND to string for one-hot encoding ---
# Map booleans to strings so that get_dummies creates distinct columns.
df['SND'] = df['SND'].str.upper().map({'TRUE': 'True', 'FALSE': 'False'})

# Print unique values in SND to ensure variability
print("Unique values in SND:", df['SND'].unique())

# --- Step 3: Compute Basic Daily Features ---
df['range'] = df['high'] - df['low']
df['body'] = abs(df['close'] - df['open'])
df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['wick_pct'] = (df['range'] - df['body']) / df['range']
df['body_ratio'] = df['body'] / df['range']

# Compute EMAs if not already present
df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
df['EMA_spread'] = df['EMA_10'] - df['EMA_50']

# Simple 14-day ATR calculation
df['tr'] = np.maximum(df['high'] - df['low'],
                      np.maximum(abs(df['high'] - df['close'].shift()),
                                 abs(df['low'] - df['close'].shift())))
df['ATR'] = df['tr'].rolling(window=14).mean()

# Create previous day hi/lo flags
df['prev_hi'] = df['high'].shift()
df['prev_lo'] = df['low'].shift()
df['hi_break'] = (df['high'] > df['prev_hi'])
df['lo_break'] = (df['low'] < df['prev_lo'])

# --- Step 4: Discretize Key Continuous Features ---
df['wick_pct_bin'] = pd.cut(df['wick_pct'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df['EMA_spread_bin'] = pd.cut(df['EMA_spread'], bins=3, labels=['NARROW','MEDIUM','WIDE'])
df['ATR_bin'] = pd.cut(df['ATR'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df['body_ratio_bin'] = pd.cut(df['body_ratio'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df['volume_bin'] = pd.cut(df['volume'], bins=3, labels=['LOW','MEDIUM','HIGH'])

# --- Step 5: Prepare Data for Association Rule Mining ---
features_for_rules = [
    'day_of_week',
    'wick_pct_bin',
    'EMA_spread_bin',
    'ATR_bin',
    'body_ratio_bin',
    'hi_break',
    'lo_break',
    'volume_bin',
    'SND'
]

df_rules = df[features_for_rules].copy()

# One-hot encode categorical variables
df_rules = pd.get_dummies(
    df_rules,
    columns=['day_of_week', 'wick_pct_bin', 'EMA_spread_bin', 'ATR_bin', 'body_ratio_bin', 'volume_bin', 'SND'],
    drop_first=False
)

# Convert hi_break and lo_break to booleans
df_rules['hi_break'] = df_rules['hi_break'].astype(bool)
df_rules['lo_break'] = df_rules['lo_break'].astype(bool)

# Convert any one-hot columns that are numeric (0/1) to booleans
for col in df_rules.columns:
    if df_rules[col].dtype == 'uint8':
        df_rules[col] = df_rules[col].astype(bool)

# Print columns to verify that SND_True exists
print("Columns in df_rules:", df_rules.columns.tolist())

# --- Step 6: Run Association Rule Mining ---
frequent_itemsets = apriori(df_rules, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Filter for rules where the consequent includes SND_True
rules_snd = rules[rules['consequents'].apply(lambda x: 'SND_True' in x)]

print("Association rules for S&D days:")
if rules_snd.empty:
    print("No rules found with these parameters.")
else:
    print(rules_snd[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# --- New Code: Convert frozensets to strings and save to SQL ---
rules_snd.loc[:, 'antecedents'] = rules_snd['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
rules_snd.loc[:, 'consequents'] = rules_snd['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

rules_snd.to_sql('daily_association_rules_v1', engine, if_exists='replace', index=False)
print("Association rules saved to 'daily_association_rules_v1' table.")

# Optional: Visualize the distribution of wick_pct
plt.figure(figsize=(10, 6))
plt.hist(df['wick_pct'].dropna(), bins=20, alpha=0.7)
plt.xlabel('Wick Percentage')
plt.ylabel('Frequency')
plt.title('Distribution of Wick Percentage')
plt.show()
