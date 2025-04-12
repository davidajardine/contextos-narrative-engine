#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import logging
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------
# Configuration
# ---------------------------
DEBUG = False         # Set to False to run on the full dataset.
SAMPLE_SIZE = 10     # Number of records to use when DEBUG is True.

# ---------------------------
# Set Up Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# 1. Data Loading
# ---------------------------
def load_aggregated_data(db_path='bybit_data.db', table_name='new_aggregated_set_1year'):
    """
    Load the one-year aggregated dataset from the specified table.
    Convert trading_day to string.
    Process the SND column robustly:
      - If SND is a string ("TRUE"/"FALSE"), map to 1/0.
      - If SND is already numeric, leave it.
    Then, create a shifted target SND_next.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    
    # Convert trading_day to string
    df['trading_day'] = pd.to_datetime(df['trading_day']).dt.date.astype(str)
    
    # Process SND column: if non-numeric, map from strings; otherwise, keep as is.
    if 'SND' in df.columns:
        if not np.issubdtype(df['SND'].dtype, np.number):
            df['SND'] = df['SND'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
    else:
        logger.warning("SND column not found in data.")
    
    # Sort by trading_day and create SND_next (shifted up by one row)
    df = df.sort_values('trading_day').reset_index(drop=True)
    df['SND_next'] = df['SND'].shift(-1)
    df = df.dropna(subset=['SND_next'])
    df['SND_next'] = df['SND_next'].astype(int)
    
    logger.info("Data loaded: %d records.", len(df))
    return df

# ---------------------------
# 2. Drop Funding Rate Columns
# ---------------------------
def drop_funding_rates(df):
    """Drop all funding_rate columns from the DataFrame."""
    cols_to_drop = ['funding_rate_A1', 'funding_rate_A2', 'funding_rate_L1', 'funding_rate_L2',
                    'funding_rate_B1', 'funding_rate_B2', 'funding_rate_P1', 'funding_rate_P2']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    logger.info("Dropped funding rate columns (if present).")
    return df

# ---------------------------
# 3. Compute Missing Metrics on the Fly
# ---------------------------
def compute_missing_metrics(df):
    """
    Compute additional metrics from available raw OHLC and range values.
    For P2:
      - top_wick_pct_P2: (high_P2 - close_P2) / range_P2
      - bottom_wick_pct_P2: (open_P2 - low_P2) / range_P2
      - liquidity_sweep_raw_P2: range_P2 / open_P2
    For A1:
      - A1_upper_wick_pct: (high_A1 - close_A1) / range_A1
      - A1_lower_wick_pct: (open_A1 - low_A1) / range_A1
      - A1_return_to_open: True if |close_A1 - open_A1|/open_A1 < 0.02
      - A1_direction: 'Bullish' if close_A1 > open_A1, else 'Bearish'
      - A1_deviation_from_P2_mid_pct: ((open_A1 - P2_midpoint) / range_P2) * 100
    Also computes P2_midpoint.
    """
    # For P2:
    df['top_wick_pct_P2'] = (df['high_P2'] - df['close_P2']) / df['range_P2']
    df['bottom_wick_pct_P2'] = (df['open_P2'] - df['low_P2']) / df['range_P2']
    df['liquidity_sweep_raw_P2'] = df['range_P2'] / df['open_P2']
    # For A1:
    df['A1_upper_wick_pct'] = (df['high_A1'] - df['close_A1']) / df['range_A1']
    df['A1_lower_wick_pct'] = (df['open_A1'] - df['low_A1']) / df['range_A1']
    df['A1_return_to_open'] = (abs(df['close_A1'] - df['open_A1']) / df['open_A1']) < 0.02
    df['A1_direction'] = np.where(df['close_A1'] > df['open_A1'], 'Bullish', 'Bearish')
    # Compute P2_midpoint and A1 deviation from P2 midpoint.
    df['P2_midpoint'] = (df['high_P2'] + df['low_P2']) / 2
    df['A1_deviation_from_P2_mid_pct'] = ((df['open_A1'] - df['P2_midpoint']) / df['range_P2']) * 100
    logger.info("Missing metrics computed.")
    return df

# ---------------------------
# 4. Binning Continuous Features
# ---------------------------
def bin_continuous_features(df, bin_config):
    """
    For each column in bin_config (dict: column -> number of bins),
    bin the continuous data into categorical labels.
    If a column is missing or has insufficient unique values, log a warning and skip.
    """
    df_binned = df.copy()
    for col, bins in bin_config.items():
        if col not in df.columns:
            logger.warning("Column '%s' not found, skipping binning.", col)
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            logger.info("Column '%s' is not numeric, skipping binning.", col)
            continue
        try:
            df_binned[col + '_bin'] = pd.qcut(df[col], q=bins,
                                              labels=[f'{col}_bin_{i+1}' for i in range(bins)],
                                              duplicates='drop')
        except Exception as e:
            logger.error("Error binning column '%s': %s", col, e)
    return df_binned

# ---------------------------
# 5. One-Hot Encoding for Categorical Columns Only
# ---------------------------
def one_hot_encode_categorical(df, additional_cols_to_keep=None):
    """
    One-hot encode only the categorical columns (object or category) in the DataFrame.
    Optionally, include additional binary columns (like SND_next).
    """
    if additional_cols_to_keep is None:
        additional_cols_to_keep = []
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_encode = cat_cols + additional_cols_to_keep
    df_encoded = pd.get_dummies(df[cols_to_encode], drop_first=False)
    return df_encoded

# ---------------------------
# 6. Run Association Rule Mining
# ---------------------------
def run_association_rule_mining(df, min_support=0.01, min_confidence=0.5, target_str='SND_next_1'):
    """
    Run Apriori on the one-hot encoded DataFrame and generate association rules.
    Filter for rules where the consequent includes the target value.
    """
    logger.info("Running association rule mining on data with %d features and %d records.", df.shape[1], len(df))
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    logger.info("Found %d frequent itemsets.", len(frequent_itemsets))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    logger.info("Generated %d association rules.", len(rules))
    rules_target = rules[rules['consequents'].apply(lambda x: target_str in x)]
    return rules_target

# ---------------------------
# 7. Save Association Rules to Database and CSV
# ---------------------------
def save_rules_to_db_and_csv(rules, db_path='bybit_data.db', table_name='new_association_rules_3h_offset', csv_filename='new_association_rules_3h_offset.csv'):
    """
    Save the association rules to an SQL table and export them as a CSV.
    Convert frozensets to comma-separated strings.
    """
    rules_copy = rules.copy()
    rules_copy['antecedents'] = rules_copy['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
    rules_copy['consequents'] = rules_copy['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
    conn = sqlite3.connect(db_path)
    rules_copy.to_sql(table_name, conn, if_exists='replace', index=False)
    rules_copy.to_csv(csv_filename, index=False)
    conn.close()
    logger.info("Association rules saved to table '%s' and CSV '%s'.", table_name, csv_filename)

# ---------------------------
# 8. Main Routine
# ---------------------------
def main():
    db_path = 'bybit_data.db'
    table_name = 'new_aggregated_set_1year'
    logger.info("Loading aggregated dataset from table '%s'...", table_name)
    df = load_aggregated_data(db_path=db_path, table_name=table_name)
    logger.info("Aggregated dataset loaded with %d records.", len(df))
    
    # Debug: Print columns and record count to ensure data is loaded
    print("Columns:", df.columns.tolist())
    print("Records loaded:", len(df))
    
    # Drop funding_rate columns
    df = drop_funding_rates(df)
    
    # Compute missing metrics on the fly
    df = compute_missing_metrics(df)
    
    # Define binning configuration (include all continuous features we want to bin)
    bin_config = {
        'daily_wick_pct': 3,
        'body_to_range_ratio': 3,
        'p2_raid_intensity': 3,
        'top_wick_pct_P2': 3,
        'bottom_wick_pct_P2': 3,
        'liquidity_sweep_raw_P2': 3,
        'A1_upper_wick_pct': 3,
        'A1_lower_wick_pct': 3,
        'A1_deviation_from_P2_mid_pct': 3,
        'EMA_5': 3,
        'EMA_100': 3,
        'EMA_350': 3,
        'ATR_7': 3
    }
    df_binned = bin_continuous_features(df, bin_config)
    sample_binned = [col for col in df_binned.columns if '_bin' in col]
    logger.info("Continuous features binned. Sample binned columns: %s", sample_binned)
    
    # For debugging, limit the sample size (adjust SAMPLE_SIZE if needed).
    if DEBUG:
        logger.info("DEBUG mode active: Using first %d records for testing.", SAMPLE_SIZE)
        df_binned = df_binned.head(SAMPLE_SIZE)
    
    # Convert SND_next to string so that one-hot encoding produces columns like SND_next_1.
    df_binned['SND_next'] = df_binned['SND_next'].astype(str)
    
    # One-hot encode only categorical columns plus the binary target SND_next.
    df_encoded = one_hot_encode_categorical(df_binned, additional_cols_to_keep=['SND_next'])
    logger.info("After one-hot encoding, the DataFrame has %d features and %d records.", df_encoded.shape[1], len(df_encoded))
    
    # Print encoded columns and sample data for verification.
    print("Encoded columns:", df_encoded.columns.tolist())
    print("Encoded data sample:")
    print(df_encoded.head())
    
    # Check unique values in each column using numpy's unique.
    unique_values = {}
    for col in df_encoded.columns:
        unique_values[col] = np.unique(df_encoded[col])
    print("Unique values per column:")
    for col, vals in unique_values.items():
        print(f"{col}: {vals}")
    
    # Run association rule mining on the one-hot encoded data.
    rules_target = run_association_rule_mining(df_encoded, min_support=0.06, min_confidence=0.5, target_str='SND_next_1')
    if rules_target.empty:
        logger.info("No association rules found with the target '%s' in the consequent.", 'SND_next_1')
    else:
        logger.info("Association rules (filtered for target='%s'):", 'SND_next_1')
        logger.info(rules_target[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))
    
    # Save the association rules to the database and CSV.
    save_rules_to_db_and_csv(rules_target, db_path=db_path, table_name='new_association_rules_3h_offset', csv_filename='new_association_rules_3h_offset.csv')
    
    logger.info("Script completed successfully on the full parameter set.")

if __name__ == "__main__":
    main()
