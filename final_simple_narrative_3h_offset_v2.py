#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
import logging
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------
# Set Up Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Data Loading Function (from new_aggregated_set_1year)
# ---------------------------
def load_aggregated_data(db_path='bybit_data.db', table_name='new_aggregated_set_1year'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    df['trading_day'] = pd.to_datetime(df['trading_day']).dt.date.astype(str)
    # Ensure SND is converted to numeric and SND_next is created
    if 'SND' in df.columns:
        df['SND'] = df['SND'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
    df = df.sort_values('trading_day').reset_index(drop=True)
    df['SND_next'] = df['SND'].shift(-1)
    df = df.dropna(subset=['SND_next'])
    df['SND_next'] = df['SND_next'].astype(int)
    return df

# ---------------------------
# Compute Missing Metrics on the Fly
# ---------------------------
def compute_missing_metrics(df):
    """
    Compute additional metrics that were not precomputed.
    Assumes the aggregated set has raw session values for P2 and A1.
    """
    # For P2:
    # Check that range_P2 is not zero to avoid division by zero.
    df['top_wick_pct_P2'] = (df['high_P2'] - df['close_P2']) / df['range_P2']
    df['bottom_wick_pct_P2'] = (df['open_P2'] - df['low_P2']) / df['range_P2']
    df['liquidity_sweep_raw_P2'] = df['range_P2'] / df['open_P2']
    # Optionally, compute volume and OI spike metrics if desired
    # For A1:
    df['A1_upper_wick_pct'] = (df['high_A1'] - df['close_A1']) / df['range_A1']
    df['A1_lower_wick_pct'] = (df['open_A1'] - df['low_A1']) / df['range_A1']
    # Compute A1_return_to_open as True if close_A1 is within 2% of open_A1
    df['A1_return_to_open'] = (abs(df['close_A1'] - df['open_A1']) / df['open_A1']) < 0.02
    # Compute A1_direction as 'Bullish' if close_A1 > open_A1, else 'Bearish'
    df['A1_direction'] = np.where(df['close_A1'] > df['open_A1'], 'Bullish', 'Bearish')
    # Compute A1 deviation from P2 midpoint
    df['P2_midpoint'] = (df['high_P2'] + df['low_P2']) / 2
    df['A1_deviation_from_P2_mid_pct'] = ((df['open_A1'] - df['P2_midpoint']) / df['range_P2']) * 100
    return df

# ---------------------------
# Binning Continuous Features
# ---------------------------
def bin_continuous_features(df, bin_config):
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
# One-Hot Encode Categorical Columns Only
# ---------------------------
def one_hot_encode_categorical(df, additional_cols_to_keep=None):
    if additional_cols_to_keep is None:
        additional_cols_to_keep = []
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_encode = cat_cols + additional_cols_to_keep
    df_encoded = pd.get_dummies(df[cols_to_encode], drop_first=False)
    return df_encoded

# ---------------------------
# Run Association Rule Mining
# ---------------------------
def run_association_rule_mining(df, min_support=0.01, min_confidence=0.5, target_str='SND_next_1'):
    logger.info("Data for association rules contains %d features and %d records.", df.shape[1], len(df))
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    logger.info("Found %d frequent itemsets.", len(frequent_itemsets))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    logger.info("Generated %d association rules.", len(rules))
    rules_target = rules[rules['consequents'].apply(lambda x: target_str in x)]
    return rules_target

# ---------------------------
# Save Association Rules
# ---------------------------
def save_rules_to_db_and_csv(rules, db_path='bybit_data.db', table_name='daily_association_rules_full', csv_filename='daily_association_rules_full.csv'):
    rules_copy = rules.copy()
    rules_copy['antecedents'] = rules_copy['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
    rules_copy['consequents'] = rules_copy['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
    conn = sqlite3.connect(db_path)
    rules_copy.to_sql(table_name, conn, if_exists='replace', index=False)
    rules_copy.to_csv(csv_filename, index=False)
    conn.close()
    logger.info("Association rules saved to table '%s' and CSV '%s'.", table_name, csv_filename)

# ---------------------------
# Main Routine
# ---------------------------
def main():
    db_path = 'bybit_data.db'
    table_name = 'new_aggregated_set_1year'
    logger.info("Loading aggregated dataset from table '%s'...", table_name)
    df = load_aggregated_data(db_path=db_path, table_name=table_name)
    logger.info("Aggregated dataset loaded with %d records.", len(df))
    
    # Compute missing metrics on the fly
    df = compute_missing_metrics(df)
    logger.info("Missing metrics computed. Sample columns added: %s", 
                [col for col in df.columns if 'top_wick_pct_P2' in col or 'A1_upper_wick_pct' in col])
    
    # Define binning configuration including all desired features
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
    
    # One-hot encode only categorical columns (plus the target if desired)
    df_encoded = one_hot_encode_categorical(df_binned, additional_cols_to_keep=['SND_next'])
    logger.info("After one-hot encoding, the DataFrame has %d features and %d records.", 
                df_encoded.shape[1], len(df_encoded))
    
    # Run association rule mining
    rules_target = run_association_rule_mining(df_encoded, min_support=0.01, min_confidence=0.5, target_str='SND_next_1')
    if rules_target.empty:
        logger.info("No association rules found with the target '%s' in the consequent.", 'SND_next_1')
    else:
        logger.info("Association rules (filtered for target='%s'):", 'SND_next_1')
        logger.info(rules_target[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))
    
    # Save rules to DB and CSV
    save_rules_to_db_and_csv(rules_target, db_path=db_path, table_name='daily_association_rules_full', csv_filename='daily_association_rules_full.csv')
    
    logger.info("Script completed successfully on the full parameter set.")

if __name__ == "__main__":
    main()
