#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging

# ---------------------------
# Set Up Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###############################
# 1. Utility Functions
###############################

def adjust_datetime(df, time_col='datetime', hours=3):
    """Shift datetime by specified hours."""
    df[time_col] = pd.to_datetime(df[time_col]) + pd.Timedelta(hours=hours)
    return df

def get_trading_day(dt):
    """
    Define trading day from 03:00 to next day 03:00.
    For times before 03:00, assign to previous day.
    """
    if dt.hour < 3:
        return (dt - timedelta(days=1)).date()
    else:
        return dt.date()

def assign_session(dt):
    """Assign session label based on adjusted time."""
    t = dt.time()
    if t >= time(0,0) and t < time(3,0):
        return 'A1'
    elif t >= time(3,0) and t < time(6,0):
        return 'A2'
    elif t >= time(6,0) and t < time(9,0):
        return 'L1'
    elif t >= time(9,0) and t < time(12,0):
        return 'L2'
    elif t >= time(12,0) and t < time(15,0):
        return 'B1'
    elif t >= time(15,0) and t < time(18,0):
        return 'B2'
    elif t >= time(18,0) and t < time(21,0):
        return 'P1'
    elif t >= time(21,0) and t <= time(23,59,59):
        return 'P2'
    else:
        return 'Other'

def compute_candle_metrics(df):
    """Compute derived metrics for a candle: range, body, upper/lower wick, wick_pct, body_ratio."""
    df = df.copy()
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['range'].replace(0, np.nan)
    df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']
    df['wick_pct'] = (df['range'] - df['body']) / df['range']
    df['body_ratio'] = df['body'] / df['range']
    return df

###############################
# 2. Data Loading & Preprocessing
###############################

def load_and_preprocess_5min(db_path='bybit_data.db'):
    """Load raw 5-min data, shift datetime, assign trading_day and session."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM btcusdt_5min", conn)
    conn.close()
    df = adjust_datetime(df, 'datetime', hours=3)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trading_day'] = df['datetime'].apply(get_trading_day)
    df['session'] = df['datetime'].apply(assign_session)
    return df

###############################
# 3. Aggregation by Session
###############################

def aggregate_session_data(df, session, agg_funcs=None):
    """
    Aggregate 5-min candles for a specific session per trading day.
    Default agg: first open, last close, max high, min low, sum volume, sum turnover, last open_interest, mean funding_rate.
    """
    df_session = df[df['session'] == session].copy()
    if agg_funcs is None:
        agg_funcs = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum',
            'open_interest': 'last',
            'funding_rate': 'mean'
        }
    agg_df = df_session.groupby('trading_day').agg(agg_funcs).reset_index()
    agg_df = compute_candle_metrics(agg_df)
    # Add session suffix to columns (except trading_day)
    agg_df = agg_df.add_suffix(f"_{session}")
    agg_df = agg_df.rename(columns={f"trading_day_{session}": "trading_day"})
    return agg_df

###############################
# 4. Create Daily Pivot Table
###############################

def create_daily_pivot(sessions_dfs, sessions):
    """Merge aggregated session data for each session into one daily row."""
    df_daily = None
    for s in sessions:
        if df_daily is None:
            df_daily = sessions_dfs[s]
        else:
            df_daily = pd.merge(df_daily, sessions_dfs[s], on='trading_day', how='outer')
    return df_daily

###############################
# 5. Compute Daily Summary Features
###############################

def compute_daily_summary(df_pivot, sessions):
    """
    Compute daily summary: daily_open, daily_close, daily_high, daily_low, daily_range,
    daily_pct_change, daily_body, body_to_range_ratio, daily_wick_pct.
    """
    df_pivot['daily_open'] = df_pivot[f"open_A2"]
    df_pivot['daily_close'] = df_pivot[f"close_A1"]
    df_pivot['daily_high'] = df_pivot[[f"high_{s}" for s in sessions]].max(axis=1)
    df_pivot['daily_low'] = df_pivot[[f"low_{s}" for s in sessions]].min(axis=1)
    df_pivot['daily_range'] = df_pivot['daily_high'] - df_pivot['daily_low']
    df_pivot['daily_pct_change'] = (df_pivot['daily_close'] - df_pivot['daily_open']) / df_pivot['daily_open']
    df_pivot['daily_body'] = (df_pivot['daily_close'] - df_pivot['daily_open']).abs()
    df_pivot['body_to_range_ratio'] = df_pivot['daily_body'] / df_pivot['daily_range']
    df_pivot['daily_wick_pct'] = (df_pivot['daily_range'] - df_pivot['daily_body']) / df_pivot['daily_range']
    return df_pivot

###############################
# 6. Compute Refined Intraday Metrics for P2 and A1
###############################

def compute_refined_P2_features(df_p2):
    """
    Compute refined P2 metrics. Expects aggregated P2 data with columns:
    high_P2, low_P2, close_P2, open_P2, body_P2, range_P2.
    """
    df = df_p2.copy()
    df['P2_midpoint'] = (df['high_P2'] + df['low_P2']) / 2
    df['P2_range'] = df['high_P2'] - df['low_P2']
    df['p2_bias_raw'] = (df['close_P2'] - df['P2_midpoint']) / df['P2_midpoint']
    df['p2_raid_intensity'] = - (df['close_P2'] - df['P2_midpoint']) / df['body_P2']
    df['top_wick_pct'] = (df['high_P2'] - df['close_P2']) / df['range_P2']
    df['bottom_wick_pct'] = (df['open_P2'] - df['low_P2']) / df['range_P2']
    df['liquidity_sweep_raw'] = df['range_P2'] / df['open_P2']
    return df

def compute_refined_A1_features(df_a1, df_p2):
    """
    Compute refined A1 metrics from aggregated A1 data.
    Expects aggregated A1 data with columns: high_A1, low_A1, close_A1, open_A1, body_A1, range_A1.
    """
    df = df_a1.copy()
    df['A1_midpoint'] = (df['high_A1'] + df['low_A1']) / 2
    df['A1_range'] = df['high_A1'] - df['low_A1']
    df['a1_bias_raw'] = (df['close_A1'] - df['A1_midpoint']) / df['A1_midpoint']
    df['a1_raid_intensity'] = - (df['close_A1'] - df['A1_midpoint']) / df['body_A1']
    df['A1_upper_wick_pct'] = (df['high_A1'] - df['close_A1']) / df['range_A1']
    df['A1_lower_wick_pct'] = (df['open_A1'] - df['low_A1']) / df['range_A1']
    # Define A1_return_to_open: True if A1 close is within 2% of A1 open
    df['A1_return_to_open'] = (abs(df['close_A1'] - df['open_A1']) / df['open_A1']) < 0.02
    # A1_direction: Bullish if close > open, else Bearish
    df['A1_direction'] = np.where(df['close_A1'] > df['open_A1'], 'Bullish', 'Bearish')
    return df

###############################
# 7. Merge Refined Features & Create Liquidity Flags
###############################

def merge_liquidity_flags(df_daily, df_p2_ext, df_a1_ext):
    """
    Merge liquidity flags into daily aggregated DataFrame.
    Expects:
      - df_daily with 'trading_day', 'daily_high', 'daily_low'
      - df_p2_ext with 'trading_day', 'high_P2', 'low_P2'
      - df_a1_ext with 'trading_day', 'high_A1', 'low_A1'
    Renames columns in extremes DataFrames and then creates flags.
    """
    df_daily['trading_day'] = df_daily['trading_day'].astype(str)
    df_p2_ext['trading_day'] = df_p2_ext['trading_day'].astype(str)
    df_a1_ext['trading_day'] = df_a1_ext['trading_day'].astype(str)
    
    df_p2_ext = df_p2_ext.rename(columns={'high_P2': 'P2_max_high', 'low_P2': 'P2_min_low'})
    df_a1_ext = df_a1_ext.rename(columns={'high_A1': 'A1_max_high', 'low_A1': 'A1_min_low'})
    
    logger.info("Columns in df_daily: %s", df_daily.columns.tolist())
    logger.info("Columns in df_p2_ext (after renaming): %s", df_p2_ext.columns.tolist())
    logger.info("Columns in df_a1_ext (after renaming): %s", df_a1_ext.columns.tolist())
    
    df_merged = pd.merge(df_daily, df_p2_ext, on='trading_day', how='left')
    df_merged = pd.merge(df_merged, df_a1_ext, on='trading_day', how='left')
    
    required_cols = ['P2_max_high', 'P2_min_low']
    for col in required_cols:
        if col not in df_merged.columns:
            raise KeyError(f"Column '{col}' is missing in the merged DataFrame.")
    
    df_merged = df_merged.sort_values('trading_day').reset_index(drop=True)
    df_merged['prev_day_high'] = df_merged['daily_high'].shift(1)
    df_merged['prev_day_low'] = df_merged['daily_low'].shift(1)
    
    df_merged['P2_tookout_daily_high'] = df_merged['P2_max_high'] > df_merged['prev_day_high']
    df_merged['P2_tookout_daily_low'] = df_merged['P2_min_low'] < df_merged['prev_day_low']
    df_merged['P2_set_daily_high'] = df_merged['P2_max_high'] == df_merged['daily_high']
    df_merged['P2_set_daily_low'] = df_merged['P2_min_low'] == df_merged['daily_low']
    
    df_merged['A1_tookout_P2_high'] = df_merged['A1_max_high'] > df_merged['P2_max_high']
    df_merged['A1_tookout_P2_low'] = df_merged['A1_min_low'] < df_merged['P2_min_low']
    df_merged['A1_tookout_daily_high'] = df_merged['A1_max_high'] > df_merged['prev_day_high']
    df_merged['A1_tookout_daily_low'] = df_merged['A1_min_low'] < df_merged['prev_day_low']
    
    return df_merged

###############################
# 8. Technical Indicators & Interaction Features
###############################

def compute_technical_indicators(df):
    """Compute EMAs, ATR, and EMA_spread on daily data."""
    df = df.sort_values('trading_day').reset_index(drop=True)
    df['EMA_5'] = df['daily_close'].ewm(span=5, adjust=False).mean()
    df['EMA_100'] = df['daily_close'].ewm(span=100, adjust=False).mean()
    df['EMA_350'] = df['daily_close'].ewm(span=350, adjust=False).mean()
    df['prev_close'] = df['daily_close'].shift(1)
    df['tr1'] = df['daily_high'] - df['daily_low']
    df['tr2'] = (df['daily_high'] - df['prev_close']).abs()
    df['tr3'] = (df['daily_low'] - df['prev_close']).abs()
    df['true_range'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['ATR_7'] = df['true_range'].rolling(window=7).mean()
    df.drop(['prev_close','tr1','tr2','tr3','true_range'], axis=1, inplace=True)
    return df

def create_interaction_features(df):
    """Create interaction features capturing combined effects."""
    # Use daily_wick_pct from daily summary multiplied by p2_raid_intensity (which must be merged in)
    df['interaction_wick_raid'] = df['daily_wick_pct'] * df['p2_raid_intensity']
    # Optionally, add an interaction for A1 deviation if present
    if 'A1_deviation_from_P2_mid_pct' in df.columns and 'P2_tookout_daily_high' in df.columns:
        df['interaction_A1_deviation_liq'] = df['A1_deviation_from_P2_mid_pct'] * df['P2_tookout_daily_high'].astype(int)
    return df

###############################
# 9. Labeling: Shift SND by One Day to Create Target (SND_next)
###############################

def shift_target_label(df):
    """Shift the SND field by one day so features from day D predict day D+1."""
    df = df.sort_values('trading_day').reset_index(drop=True)
    df['SND_next'] = df['SND'].shift(-1)
    df = df.dropna(subset=['SND_next'])
    df['SND_next'] = df['SND_next'].astype(int)
    return df

###############################
# 10. Save Final Dataset
###############################

def save_final_dataset(df, table_name='new_aggregated_set', csv_filename='new_aggregated_set.csv', db_path='bybit_data.db'):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    df.to_csv(csv_filename, index=False)
    conn.close()
    logger.info(f"Final aggregated dataset saved to table '{table_name}' and CSV '{csv_filename}'.")

###############################
# 11. Main Routine: Build New Aggregated Set
###############################

def main():
    db_path = 'bybit_data.db'
    
    logger.info("Loading raw 5-min data...")
    df_5min = load_and_preprocess_5min(db_path=db_path)
    logger.info(f"5-min data loaded: {len(df_5min)} rows.")
    
    # Aggregate session data for each session
    sessions = ['A1', 'A2', 'L1', 'L2', 'B1', 'B2', 'P1', 'P2']
    sessions_dfs = {}
    for s in sessions:
        sessions_dfs[s] = aggregate_session_data(df_5min, s)
        logger.info(f"Aggregated session {s}: {len(sessions_dfs[s])} rows.")
    
    # Create daily pivot: one row per trading day with session-level metrics
    df_pivot = create_daily_pivot(sessions_dfs, sessions)
    logger.info(f"Daily pivot created with {len(df_pivot)} rows.")
    
    # Compute daily summary metrics
    df_daily_agg = compute_daily_summary(df_pivot, sessions)
    logger.info("Daily summary metrics computed.")
    
    # Compute refined P2 features from aggregated P2 data (from sessions_dfs['P2'])
    df_P2_refined = compute_refined_P2_features(sessions_dfs['P2'])
    # Compute refined A1 features from aggregated A1 data (from sessions_dfs['A1'])
    df_A1_refined = compute_refined_A1_features(sessions_dfs['A1'], sessions_dfs['P2'])
    
    # For liquidity flags, extract extremes:
    df_P2_ext = sessions_dfs['P2'][['trading_day', 'high_P2', 'low_P2']].groupby('trading_day').agg({
        'high_P2': 'max',
        'low_P2': 'min'
    }).reset_index()
    df_A1_ext = sessions_dfs['A1'][['trading_day', 'high_A1', 'low_A1']].groupby('trading_day').agg({
        'high_A1': 'max',
        'low_A1': 'min'
    }).reset_index()
    
    # Merge the refined P2 features into the daily aggregated dataset
    df_merged = df_daily_agg.copy()
    # Merge in refined P2 features (specifically, p2_raid_intensity)
    df_merged = pd.merge(df_merged, df_P2_refined[['trading_day', 'p2_raid_intensity']], on='trading_day', how='left')
    
    # Merge liquidity target flags using the extremes data for P2 and A1
    df_merged = merge_liquidity_flags(df_merged, df_P2_ext, df_A1_ext)
    logger.info("Liquidity target flags merged.")
    
    # Merge technical indicators
    df_merged = compute_technical_indicators(df_merged)
    logger.info("Technical indicators computed.")
    
    # Create interaction features
    df_merged = create_interaction_features(df_merged)
    
    # Merge SND target from btcusdt_daily
    conn = sqlite3.connect(db_path)
    btc_daily = pd.read_sql_query("SELECT datetime, SND FROM btcusdt_daily", conn)
    conn.close()
    btc_daily['datetime'] = pd.to_datetime(btc_daily['datetime'])
    btc_daily.sort_values('datetime', inplace=True)
    btc_daily['date'] = btc_daily['datetime'].dt.date
    # Merge on trading_day (ensure type consistency)
    df_merged['trading_day'] = df_merged['trading_day'].astype(str)
    btc_daily['date'] = btc_daily['date'].astype(str)
    df_merged = pd.merge(df_merged, btc_daily[['date', 'SND']], left_on='trading_day', right_on='date', how='left')
    df_merged.drop(columns=['date'], inplace=True)
    df_merged['SND'] = df_merged['SND'].apply(lambda x: 1 if str(x).strip().upper() in ["TRUE", "1"] else 0)
    
    # Shift SND to create SND_next
    df_merged = shift_target_label(df_merged)
    logger.info("Target label (SND_next) created by shifting SND.")
    
    # Save the final aggregated dataset
    save_final_dataset(df_merged, table_name='new_aggregated_set', csv_filename='new_aggregated_set.csv', db_path=db_path)
    logger.info("New aggregated dataset built and saved successfully.")

if __name__ == "__main__":
    main()
