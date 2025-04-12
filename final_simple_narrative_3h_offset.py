import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules

print("Starting production script with SND offset...")

# ---------------------------
# Define Database Engine
# ---------------------------
engine = create_engine('sqlite:///bybit_data.db')

# ---------------------------
# 1. Load Daily Data, Compute Daily Candle Metrics, and Create SND_next (Outcome)
# ---------------------------
df_daily = pd.read_sql("SELECT * FROM btcusdt_daily_1year", engine)
df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
df_daily['trading_day'] = df_daily['datetime'].dt.date

# Convert SND values to numeric: 1 for TRUE, 0 for FALSE (assuming stored as text)
df_daily['SND'] = df_daily['SND'].str.upper().map({'TRUE': 1, 'FALSE': 0})

# Debug: Check SND distribution
print("SND value counts in daily data:")
print(df_daily['SND'].value_counts())

# Compute daily candle metrics
df_daily['range'] = df_daily['high'] - df_daily['low']
df_daily['body'] = abs(df_daily['close'] - df_daily['open'])
df_daily['range'] = df_daily['range'].replace(0, np.nan)
df_daily['wick_pct'] = (df_daily['range'] - df_daily['body']) / df_daily['range']
df_daily['body_ratio'] = df_daily['body'] / df_daily['range']

# Sort daily data and shift SND upward by one row (so features from day N predict day N+1)
df_daily = df_daily.sort_values('trading_day').reset_index(drop=True)
df_daily['SND_next'] = df_daily['SND'].shift(-1)
df_daily = df_daily.dropna(subset=['SND_next'])
df_daily['SND_next'] = df_daily['SND_next'].astype(int)

print("SND_next value counts (target):")
print(df_daily['SND_next'].value_counts())

# ---------------------------
# 2. Load 3‑h Data & Assign Session Labels
# ---------------------------
df_3h = pd.read_sql("SELECT * FROM btcusdt_3h_1year", engine)
df_3h['datetime'] = pd.to_datetime(df_3h['datetime'])
df_3h['trading_day'] = df_3h['datetime'].dt.date

def assign_session(dt):
    t = dt.time()
    if t >= pd.to_datetime("00:00").time() and t < pd.to_datetime("03:00").time():
        return 'A1'
    elif t >= pd.to_datetime("03:00").time() and t < pd.to_datetime("06:00").time():
        return 'A2'
    elif t >= pd.to_datetime("06:00").time() and t < pd.to_datetime("09:00").time():
        return 'L1'
    elif t >= pd.to_datetime("09:00").time() and t < pd.to_datetime("12:00").time():
        return 'L2'
    elif t >= pd.to_datetime("12:00").time() and t < pd.to_datetime("15:00").time():
        return 'B1'
    elif t >= pd.to_datetime("15:00").time() and t < pd.to_datetime("18:00").time():
        return 'B2'
    elif t >= pd.to_datetime("18:00").time() and t < pd.to_datetime("21:00").time():
        return 'P1'
    elif t >= pd.to_datetime("21:00").time() and t <= pd.to_datetime("23:59:59").time():
        return 'P2'
    else:
        return 'Other'

df_3h['session'] = df_3h['datetime'].apply(assign_session)
print("3‑h data loaded and sessions assigned.")

# ---------------------------
# 3. Compute Candle Metrics for 3‑h Data (Common Function)
# ---------------------------
def compute_candle_metrics(df):
    df = df.copy()
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['range'].replace(0, np.nan)
    df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']
    df['wick_pct'] = (df['range'] - df['body']) / df['range']
    df['body_ratio'] = df['body'] / df['range']
    return df

df_P1 = compute_candle_metrics(df_3h[df_3h['session'] == 'P1'])
df_P2 = compute_candle_metrics(df_3h[df_3h['session'] == 'P2'])
df_A1 = compute_candle_metrics(df_3h[df_3h['session'] == 'A1'])
print("Computed candle metrics for P1, P2, and A1 sessions.")

# ---------------------------
# 4. Compute 3‑h Specific Metrics for P2
# ---------------------------
df_P2 = df_P2.copy()
df_P2['midpoint'] = (df_P2['high'] + df_P2['low']) / 2
df_P2['p2_bias_raw'] = (df_P2['close'] - df_P2['midpoint']) / df_P2['midpoint']
df_P2['p2_raid_intensity'] = - (df_P2['close'] - df_P2['midpoint']) / df_P2['body']
df_P2['top_wick_pct'] = (df_P2['high'] - df_P2['close']) / df_P2['range']
df_P2['bottom_wick_pct'] = (df_P2['open'] - df_P2['low']) / df_P2['range']
df_P2['liquidity_sweep_raw'] = df_P2['range'] / df_P2['open']
benchmark_vol_P2 = df_P2['volume'].mean()
benchmark_oi_P2 = df_P2['open_interest'].mean()
df_P2['volume_spike_raw'] = df_P2['volume'] / benchmark_vol_P2
df_P2['oi_spike_raw'] = df_P2['open_interest'] / benchmark_oi_P2
print("P2 specific metrics computed.")

# ---------------------------
# 5. Compute Technical Indicators on 3‑h Data (EMA, ATR)
# ---------------------------
df_3h['EMA_10'] = df_3h['close'].ewm(span=10, adjust=False).mean()
df_3h['EMA_50'] = df_3h['close'].ewm(span=50, adjust=False).mean()
df_3h['EMA_spread'] = df_3h['EMA_10'] - df_3h['EMA_50']
print("EMA indicators computed on 3‑h data.")

# ---------------------------
# 6. Aggregate 3‑h Metrics by Trading Day
# ---------------------------
agg_P2 = df_P2.groupby('trading_day').agg({
    'p2_bias_raw': 'mean',
    'p2_raid_intensity': 'mean',
    'top_wick_pct': 'mean',
    'bottom_wick_pct': 'mean',
    'liquidity_sweep_raw': 'mean',
    'volume_spike_raw': 'max',
    'oi_spike_raw': 'max'
}).reset_index()

agg_P1 = df_P1.groupby('trading_day').agg({
    'wick_pct': 'mean',
    'body_ratio': 'mean'
}).reset_index().rename(columns={'wick_pct': 'P1_wick_pct', 'body_ratio': 'P1_body_ratio'})

agg_A1 = df_A1.groupby('trading_day').agg({
    'wick_pct': 'mean',
    'body_ratio': 'mean'
}).reset_index().rename(columns={'wick_pct': 'A1_wick_pct', 'body_ratio': 'A1_body_ratio'})

agg_EMA = df_3h.groupby('trading_day').agg({
    'EMA_spread': 'mean'
}).reset_index()

print("Aggregated 3‑h metrics by trading day.")

# ---------------------------
# 7. Merge Aggregated 3‑h Metrics with Daily Data
# ---------------------------
daily_merged = df_daily.copy()
daily_merged['trading_day'] = pd.to_datetime(daily_merged['datetime']).dt.date
daily_merged = daily_merged.merge(agg_P2, on='trading_day', how='left')
daily_merged = daily_merged.merge(agg_P1, on='trading_day', how='left')
daily_merged = daily_merged.merge(agg_A1, on='trading_day', how='left')
daily_merged = daily_merged.merge(agg_EMA, on='trading_day', how='left')
print("Merged daily data now has", len(daily_merged), "records.")

# ---------------------------
# 8. Add Additional Liquidity Target Flags for P2
# ---------------------------
# Compute extremes for P1 and P2 from 3‑h data
agg_P1_ext = df_3h[df_3h['session'] == 'P1'].groupby('trading_day').agg({
    'high': 'max',
    'low': 'min'
}).reset_index().rename(columns={'high': 'P1_high', 'low': 'P1_low'})

agg_P2_ext = df_P2.groupby('trading_day').agg({
    'high': 'max',
    'low': 'min'
}).reset_index().rename(columns={'high': 'P2_max_high', 'low': 'P2_min_low'})

# Merge extremes into daily_merged
daily_merged = daily_merged.merge(agg_P1_ext, on='trading_day', how='left')
daily_merged = daily_merged.merge(agg_P2_ext, on='trading_day', how='left')

# Compute previous day's high and low from daily data
daily_merged['prev_day_high'] = daily_merged['high'].shift(1)
daily_merged['prev_day_low'] = daily_merged['low'].shift(1)

# Create liquidity target flags:
daily_merged['P2_tookout_daily_high'] = daily_merged['P2_max_high'] > daily_merged['prev_day_high']
daily_merged['P2_tookout_daily_low'] = daily_merged['P2_min_low'] < daily_merged['prev_day_low']
daily_merged['P2_set_daily_high'] = daily_merged['P2_max_high'] == daily_merged['high']
daily_merged['P2_set_daily_low'] = daily_merged['P2_min_low'] == daily_merged['low']

print("Refined P2 metrics added. Sample:")
print(daily_merged[['trading_day', 'SND', 'wick_pct', 'p2_raid_intensity', 
                      'P2_tookout_daily_high', 'P2_tookout_daily_low',
                      'P2_set_daily_high', 'P2_set_daily_low']].head())

# ---------------------------
# 9. Create Interaction Features (Optional)
# ---------------------------
daily_merged['interaction_wick_raid'] = daily_merged['wick_pct'] * daily_merged['p2_raid_intensity']

# ---------------------------
# 10. Prepare Enriched Daily Data for Association Rule Mining
# ---------------------------
if 'day_of_week' not in daily_merged.columns:
    daily_merged['day_of_week'] = pd.to_datetime(daily_merged['datetime']).dt.day_name()

features_for_rules = [
    'day_of_week',           # categorical from daily data
    'wick_pct',              # daily candle metric
    'body_ratio',            # daily candle metric
    'p2_raid_intensity',     # aggregated from P2 metrics
    'top_wick_pct',          # aggregated from P2 metrics
    'bottom_wick_pct',       # aggregated from P2 metrics
    'liquidity_sweep_raw',   # aggregated from P2 metrics
    'volume_spike_raw',      # aggregated from P2 metrics
    'oi_spike_raw',          # aggregated from P2 metrics
    'EMA_spread',            # aggregated from 3h EMA spread
    'P2_tookout_daily_high',
    'P2_tookout_daily_low',
    'P2_set_daily_high',
    'P2_set_daily_low',
    'SND_next'               # Use the shifted SND as the target outcome
]

df_enriched = daily_merged[features_for_rules].copy()

# ---------------------------
# 11. Discretize Continuous Features for Association Mining
# ---------------------------
df_enriched['wick_pct_bin'] = pd.cut(df_enriched['wick_pct'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df_enriched['body_ratio_bin'] = pd.cut(df_enriched['body_ratio'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df_enriched['p2_raid_bin'] = pd.cut(df_enriched['p2_raid_intensity'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df_enriched['top_wick_bin'] = pd.cut(df_enriched['top_wick_pct'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df_enriched['bottom_wick_bin'] = pd.cut(df_enriched['bottom_wick_pct'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df_enriched['liq_sweep_bin'] = pd.cut(df_enriched['liquidity_sweep_raw'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df_enriched['vol_spike_bin'] = pd.cut(df_enriched['volume_spike_raw'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df_enriched['oi_spike_bin'] = pd.cut(df_enriched['oi_spike_raw'], bins=3, labels=['LOW','MEDIUM','HIGH'])
df_enriched['EMA_spread_bin'] = pd.cut(df_enriched['EMA_spread'], bins=3, labels=['NARROW','MEDIUM','WIDE'])

# Retain day_of_week and SND_next as is (SND_next is numeric, so convert to string for one-hot encoding)
df_enriched['SND_next'] = df_enriched['SND_next'].astype(str)
# Convert liquidity target flags to strings
df_enriched['P2_tookout_daily_high'] = df_enriched['P2_tookout_daily_high'].astype(str)
df_enriched['P2_tookout_daily_low'] = df_enriched['P2_tookout_daily_low'].astype(str)
df_enriched['P2_set_daily_high'] = df_enriched['P2_set_daily_high'].astype(str)
df_enriched['P2_set_daily_low'] = df_enriched['P2_set_daily_low'].astype(str)

df_enriched = df_enriched[[
    'day_of_week',
    'wick_pct_bin',
    'body_ratio_bin',
    'p2_raid_bin',
    'top_wick_bin',
    'bottom_wick_bin',
    'liq_sweep_bin',
    'vol_spike_bin',
    'oi_spike_bin',
    'EMA_spread_bin',
    'P2_tookout_daily_high',
    'P2_tookout_daily_low',
    'P2_set_daily_high',
    'P2_set_daily_low',
    'SND_next'
]].copy()

# ---------------------------
# 12. One-Hot Encode and Run Association Rule Mining
# ---------------------------
df_rules = pd.get_dummies(df_enriched, columns=[
    'day_of_week', 'wick_pct_bin', 'body_ratio_bin', 'p2_raid_bin',
    'top_wick_bin', 'bottom_wick_bin', 'liq_sweep_bin', 'vol_spike_bin',
    'oi_spike_bin', 'EMA_spread_bin', 'P2_tookout_daily_high',
    'P2_tookout_daily_low', 'P2_set_daily_high', 'P2_set_daily_low', 'SND_next'
], drop_first=False)

print("Data for association rules contains", df_rules.shape[1], "features and", len(df_rules), "records.")

frequent_itemsets = apriori(df_rules, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Filter for rules with SND_next_1 in the consequent (target outcome is SND_next=1)
rules_snd = rules[rules['consequents'].apply(lambda x: 'SND_next_1' in x)]

print("Association rules for S&D days (offset target):")
if rules_snd.empty:
    print("No rules found with these parameters.")
else:
    print(rules_snd[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ---------------------------
# 13. Save Association Rules to New Table 'daily_association_rules_3h'
# ---------------------------
# Convert frozensets to comma-separated strings for saving
rules_snd.loc[:, 'antecedents'] = rules_snd['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
rules_snd.loc[:, 'consequents'] = rules_snd['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

rules_snd.to_sql('daily_association_rules_3h', engine, if_exists='replace', index=False)
print("Association rules saved to table 'daily_association_rules_3h'.")

# ---------------------------
# 14. Save Enriched Daily Dataset to CSV
# ---------------------------
daily_merged.to_csv('daily_merged_with_3h_features.csv', index=False)
print("Enriched daily dataset saved to 'daily_merged_with_3h_features.csv'.")
print("Script completed successfully!")
