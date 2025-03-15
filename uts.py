import pandas as pd
import os
import argparse
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(fama_data_path, filing_data_path, pre_event_data_path, post_event_data_path):
   

    try:
        fama_data = pd.read_csv(fama_data_path) #Always the same
        logging.info(f"Fama-French data loaded successfully.")
        if fama_data.empty:
            logging.error(f"Fama-French data at {fama_data_path} is empty.")
            raise ValueError(f"Fama-French data at {fama_data_path} is empty.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Fama-French data file not found at {fama_data_path}.")
    except ValueError as ve:
        raise ve
    
    try:
        pre_event_data = pd.read_csv(pre_event_data_path)
        logging.info(f"Pre Event Regression data loaded successfully.")
        if pre_event_data.empty:
            logging.error("Pre Event regression data is empty.")
            raise ValueError
    except FileNotFoundError:
        raise FileNotFoundError("Pre Event Regression file not found.")
    except ValueError as ve:
        raise ve
    
    try: 
        post_event_data = pd.read_csv(post_event_data_path)
        logging.info("Post Event Regression data loaded successfully.")
        if post_event_data.empty:
            logging.error("Post Event Regression data is empty.")
            raise ValueError
    except FileNotFoundError:
        raise FileNotFoundError("Post Event Regression file not found.")
    except ValueError as ve:
        raise ve
    
    try:
        filing_data = pd.read_parquet(filing_data_path)
        logging.info("Filing Data loaded successfully.")
        if filing_data.empty:
            logging.error("Filing Data is empty")
            raise ValueError
    except FileNotFoundError:
        raise FileNotFoundError("Filing Data file not found.")
    except ValueError as ve:
        raise ve
    
    return fama_data, pre_event_data, post_event_data, filing_data

def add_filing_date(pre_event_data, post_event_data, filing_data):

    # Ensure inputs are DataFrames
    if not isinstance(pre_event_data, pd.DataFrame) or not isinstance(post_event_data, pd.DataFrame):
        raise TypeError("Pre Event data and Post Event data must be a DataFrame.")
    if not isinstance(filing_data, pd.DataFrame):
        raise TypeError("Filing Data must be a DataFrame")

    # Copy DataFrames to avoid modifying originals
    pre_df = pre_event_data.copy()
    post_df = post_event_data.copy()
    filing_df = filing_data.copy()

    # Ensure required columns exist
    for df, name in zip([pre_df, post_df, filing_df], ["Pre Event", "Post Event", "Filing"]):
        if "CIK" not in df.columns:
            raise ValueError(f"Missing 'CIK' column in {name} data.")
    if "Filing Date" not in filing_df.columns:
        raise ValueError("Missing 'Filing Date' column in filing data.")

    # Convert 'CIK' to string format to prevent merge errors
    pre_df["CIK"] = pre_df["CIK"].astype(str)
    post_df["CIK"] = post_df["CIK"].astype(str)
    filing_df["CIK"] = filing_df["CIK"].astype(str)

    # Convert Filing Date to datetime format
    filing_df["Filing Date"] = pd.to_datetime(filing_df["Filing Date"], errors="coerce")

    # Merge filing date into pre-event and post-event datasets
    merged_pre = pd.merge(filing_df, pre_df, on="CIK", how="inner")
    merged_post = pd.merge(filing_df, post_df, on="CIK", how="inner")

    # Ensure column order
    column_order = ["Filing Date", "CIK"] + [col for col in pre_df.columns if col != "CIK"]
    merged_pre = merged_pre[column_order]
    merged_post = merged_post[column_order]

    return merged_pre, merged_post
