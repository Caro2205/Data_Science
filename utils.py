import pandas as pd
from datetime import datetime, timedelta
import numpy as np  
import math
import statsmodels.api as sm
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(fama_data_path, cik_path, ric_path):
    '''loads all necessary files for preprocessing
    input = folder containing stock files per year
            fama-french factors file
            cik and ric file per year
    output = one parquet file per year and quarter merged with fama file'''

    try:
        fama_data = pd.read_csv(fama_data_path) # Always the same
        logging.info(f"Fama-French data loaded successfully.")
        if fama_data.empty:
            logging.error(f"Fama-French data at {fama_data_path} is empty.")
            raise ValueError(f"Fama-French data at {fama_data_path} is empty.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Fama-French data file not found at {fama_data_path}.")
    except ValueError as ve:
        raise ve

    try:
        cik_data = pd.read_parquet(cik_path) # Used for the filing date
        logging.info(f"CIK data loaded successfully.")
        if cik_data.empty:
            logging.error(f"CIK data at {cik_path} is empty.")
            raise ValueError(f"CIK data at {cik_path} is empty.")
    except FileNotFoundError:
        raise FileNotFoundError(f"CIK data file not found at {cik_path}.")
    except ValueError as ve:
        raise ve

    try:
        ric_data = pd.read_parquet(ric_path) # Used for the mapping
        logging.info(f"RIC data loaded successfully.")
        if ric_data.empty:
            logging.error(f"RIC data at {ric_path} is empty.")
            raise ValueError(f"RIC data at {ric_path} is empty.")
    except FileNotFoundError:
        raise FileNotFoundError(f"RIC data file not found at {ric_path}.")
    except ValueError as ve:
        raise ve

    return fama_data, cik_data, ric_data

def merge_stock_data(stock_data, post_stock_data, mapping_data):
    '''Merges stock data with mapping data (CIK)'''

    if not isinstance(stock_data, pd.DataFrame) or not isinstance(post_stock_data, pd.DataFrame):
        raise TypeError("stock_data, and post_stock_data must be a DataFrame")
    if not isinstance(mapping_data, pd.DataFrame):
        raise TypeError("mapping_data must be a DataFrame")
    
    stock_df = stock_data.copy()
    post_stock_df = post_stock_data.copy()
    mapping_df = mapping_data.copy()

    stock_df["Date"] = stock_df["Date"].astype(str).str[:10]  # Ensure Date is string before slicing
    stock_df["Price Close"] = pd.to_numeric(stock_df["Price Close"], errors='coerce')  # Convert to numeric

    #Post Event Data
    post_stock_df["Date"] = post_stock_df["Date"].astype(str).str[:10]  
    post_stock_df["Price Close"] = pd.to_numeric(post_stock_df["Price Close"], errors='coerce')

    # Clean RIC column
    stock_df["Clean RIC"] = stock_df["RIC"].str.split("^").str[0]  
    post_stock_df["Clean RIC"] = post_stock_df["RIC"].str.split("^").str[0]

    dropped_duplicates = mapping_df.duplicated(subset="RIC_clean").sum()
    mapping_df = mapping_df.drop_duplicates(subset="RIC_clean", keep="first")
    logging.info(f"Removed {dropped_duplicates} duplicates from dataset")


    merged_data = stock_df.merge(mapping_df[["RIC_clean", "cik"]],
                                 left_on="Clean RIC", right_on="RIC_clean", how="left")
    
    merged_post_data = post_stock_df.merge(mapping_df[["RIC_clean", "cik"]],
                                    left_on="Clean RIC", right_on="RIC_clean", how="left")

    final_data = merged_data[["Date", "Price Close", "Clean RIC", "cik"]].rename(columns={"cik": "CIK"})
    final_post_data = merged_post_data[["Date", "Price Close", "Clean RIC", "cik"]].rename(columns={"cik": "CIK"})

    return final_data, final_post_data


def organize_and_calculate_returns(merged_data, merged_post_data):

    if not isinstance(merged_data, pd.DataFrame) or not isinstance(merged_post_data, pd.DataFrame):
        raise TypeError("merged_data and merged_post_data must be DataFrames")
    
    stock_df = merged_data.copy()
    post_stock_df = merged_post_data.copy()

    # Convert to Datetime
    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    post_stock_df["Date"] = pd.to_datetime(post_stock_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Sort by CIK and Date
    stock_df = stock_df.sort_values(by=["CIK", "Date"]).reset_index(drop=True)
    post_stock_df = post_stock_df.sort_values(by=["CIK", "Date"]).reset_index(drop=True)

    # Calculate daily returns
    stock_df["Daily Returns"] = stock_df.groupby("CIK")["Price Close"].pct_change().fillna(0) * 100
    post_stock_df["Daily Returns"] = post_stock_df.groupby("CIK")["Price Close"].pct_change(fill_method=None).fillna(0) * 100

    return stock_df, post_stock_df  


def add_filing_date(daily_returns, post_daily_returns, cik_data): 

    if not isinstance(daily_returns, pd.DataFrame):
        raise TypeError("daily returns must be DataFrames")
    elif not isinstance(post_daily_returns, pd.DataFrame):
        raise TypeError("post_daily_returns must be a DataFrame")
    elif not isinstance(cik_data, pd.DataFrame):
        raise TypeError("cik_data must be a DataFrame")
    
    daily_df = daily_returns.copy()
    post_daily_df = post_daily_returns.copy()

    #Make a copy of the cik_data for the post event analysis
    filing_date_df = cik_data.copy()
    post_filing_date_df = cik_data.copy()

    # Clean the extra 0 in full date
    filing_date_df["full_date"] = filing_date_df["full_date"].astype(str).str[:10]
    post_filing_date_df["full_date"] = post_filing_date_df["full_date"].astype(str).str[:10]

    #Sort by CIK and Date
    filing_date_df = filing_date_df.sort_values(by=["cik", "full_date"]).reset_index(drop=True)
    post_filing_date_df = post_filing_date_df.sort_values(by=["cik", "full_date"]).reset_index(drop=True)

    #Convert to datetime
    filing_date_df["full_date"] = pd.to_datetime(filing_date_df["full_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    post_filing_date_df["full_date"] = pd.to_datetime(post_filing_date_df["full_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    
    #remove zero padding
    daily_df["CIK"] = daily_df["CIK"].astype(int).astype(str).str.zfill(10)
    post_daily_df["CIK"] = post_daily_df["CIK"].astype(int).astype(str).str.zfill(10)

    filing_date_df["cik"] = filing_date_df["cik"].astype(str).str.zfill(10)
    post_filing_date_df["cik"] = post_filing_date_df["cik"].astype(str).str.zfill(10)

    # Extract filing date for pre event = max date
    latest_filing_dates = filing_date_df.groupby("cik")["full_date"].max().reset_index()
    latest_filing_dates.rename(columns={"full_date": "Filing Date"}, inplace=True)

    #Extract filing date for post event = min date
    # Filter dates within the range 2015-2024
    valid_dates = post_filing_date_df[(post_filing_date_df["full_date"] >= "2015-01-01") & (post_filing_date_df["full_date"] <= "2024-12-31")]

    if valid_dates.empty:
        logging.warning("No valid filing dates found within the range 2015-2024.")
        early_filing_dates = pd.DataFrame(columns=["cik", "Filing Date"])
    else:
        early_filing_dates = valid_dates.groupby("cik")["full_date"].min().reset_index()
        early_filing_dates.rename(columns={"full_date": "Filing Date"}, inplace=True)

    # Log excluded CIKs
    excluded_ciks = post_filing_date_df[~post_filing_date_df["cik"].isin(early_filing_dates["cik"])]
    if not excluded_ciks.empty:
        logging.warning(f"Excluded CIKs with dates outside 2015-2024 range: {excluded_ciks['cik'].unique()}")

    # Merging
    merged_data = daily_df.merge(latest_filing_dates, left_on="CIK", right_on="cik", how="left")
    merged_data = merged_data.drop(columns=["cik"])

    post_merged_data = post_daily_df.merge(early_filing_dates, left_on="CIK", right_on="cik", how="left")
    post_merged_data = post_merged_data.drop(columns=["cik"])

    #Check if any filing dates were matched
    missing_filing_dates = merged_data["Filing Date"].isnull().sum()
    post_missing_filing_dates = post_merged_data["Filing Date"].isnull().sum()

    if missing_filing_dates > 0:
        logging.warning(f"Warning: {missing_filing_dates} rows have missing Filing Date!")

    if post_missing_filing_dates > 0:
        logging.warning(f"Warning: {post_missing_filing_dates} rows have missing Filing Date!")

    return merged_data, post_merged_data


def add_fama_french_factors(merged_data, post_merged_data, fama_data):

    if not isinstance(merged_data, pd.DataFrame) or not isinstance(post_merged_data, pd.DataFrame) or not isinstance(fama_data, pd.DataFrame):
        raise TypeError("Inputs must be DataFrames")
    
    fama_df = fama_data.copy()
    fama_df.drop(columns=["RMW", "CMA"], errors="ignore", inplace=True)

    merged_df = merged_data.merge(fama_df, on="Date", how="left")
    post_merged_df = post_merged_data.merge(fama_df, on="Date", how="left")
    
    return merged_df, post_merged_df

def perform_fama_french_regression(stock_data, output_folder, year):
    if not isinstance(stock_data, pd.DataFrame):
        raise TypeError("stock_data must be a DataFrame")

    stock_data["Filing Date"] = pd.to_datetime(stock_data["Filing Date"])
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])  # Update to daily granularity

    time_windows = {250: 250}  # Define windows in days

    results = []
    excluded_ciks = []
    
    for cik, group in stock_data.groupby("CIK"):
        filing_date = group["Filing Date"].max()

        for window, days in time_windows.items():
            start_date = filing_date - pd.Timedelta(days=days)
            regression_data = group[(group["Date"] >= start_date) & (group["Date"] < filing_date)].copy()

            if regression_data.empty or len(regression_data) < 5:
                excluded_ciks.append({"CIK": cik, "Window": window, "Reason": "Insufficient data"})
                continue

            if "Daily Returns" not in regression_data or "RF" not in regression_data:
                excluded_ciks.append({"CIK": cik, "Window": window, "Reason": "Missing required columns"})
                continue

            regression_data["Excess Return"] = regression_data["Daily Returns"] - regression_data["RF"]

            X = regression_data[["Mkt-RF", "SMB", "HML"]]
            X = sm.add_constant(X)
            y = regression_data["Excess Return"]

            X, y = X.dropna(), y.loc[X.index]
            if X.empty or y.empty:
                excluded_ciks.append({"CIK": cik, "Window": window, "Reason": "Insufficient valid data after dropna()"})
                continue

            try:
                model = sm.OLS(y, X).fit()
                if model.rsquared is None or np.isnan(model.rsquared) or np.isinf(model.rsquared):
                    raise ValueError("Invalid Regression Results (R squared is NaN or inf)")
                else:
                    results.append({
                    "CIK": cik,
                    "Window": window,
                    "Alpha": round(model.params.get("const", None), 4),
                    "Beta_Mkt-RF": round(model.params.get("Mkt-RF", None), 4) if model.params.get("Mkt-RF", None) is not None else None,
                    "Beta_SMB": round(model.params.get("SMB", None), 4) if model.params.get("SMB", None) is not None else None,
                    "Beta_HML": round(model.params.get("HML", None), 4) if model.params.get("HML", None) is not None else None,
                    "R-squared": round(model.rsquared, 4)
                })
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
                excluded_ciks.append({"CIK": cik, "Window": window, "Reason": f"Regression failure: {str(e)}"})
                continue
    
    # Save excluded CIKs to CSV
    if excluded_ciks:
        excluded_df = pd.DataFrame(excluded_ciks)
        excluded_path = os.path.join(output_folder, f"Excluded_CIKs_{year}.csv")
        excluded_df.to_csv(excluded_path, index=False)
        logging.info(f"Excluded CIKs saved to {excluded_path}")
    
    return pd.DataFrame(results)


def perform_post_regression(post_stock_data, output_folder, year):

    if not isinstance(post_stock_data, pd.DataFrame):
        raise TypeError("post_stock_data must be a DataFrame")
    
    post_stock_data["Filing Date"] = pd.to_datetime(post_stock_data["Filing Date"])
    post_stock_data["Date"] = pd.to_datetime(post_stock_data["Date"])

    
    timewindows = {30: 30,
                   60: 60,
                   90: 90,
                   "1_day_before_after": 1}
    
    
    n_min = 5
    max_days = 30

    datapoints = post_stock_data.groupby("CIK").size

    results = []
    excluded_ciks = []

    for cik, group in post_stock_data.groupby("CIK"):
        filing_date = group["Filing Date"].min()

        for window, days in timewindows.items():
            if window == "1_day_before_after":

                dynamic_days = days
                while dynamic_days <= max_days:
                    start_date = filing_date - pd.Timedelta(days=dynamic_days)
                    end_date = filing_date + pd.Timedelta(days=dynamic_days)

                    regression_data = group[(group["Date"] >= start_date) &(group["Date"] <= end_date)].copy()

                    if len(regression_data) >= n_min:
                        break
                
                    dynamic_days += 1

                if len(regression_data) < n_min:
                    excluded_ciks.append({"CIK":cik, "Window":window, "Reason": "Insufficient Data after Expansion"})
                    continue


            else:
                start_date = filing_date
                end_date = filing_date + pd.Timedelta(days=days)
            regression_data = group[(group["Date"] >= start_date) & (group["Date"] <= end_date)].copy()
        
            if regression_data.empty or len(regression_data) < 5:
                excluded_ciks.append({"CIK": cik, "Window": window, "Reason": "Insufficient data"})
                continue

            if "Daily Returns" not in regression_data or "RF" not in regression_data:
                excluded_ciks.append({"CIK": cik, "Window": window, "Reason": "Missing required columns"})
                continue


            regression_data["Excess Return"] = regression_data["Daily Returns"] - regression_data["RF"]

            X = regression_data[["Mkt-RF", "SMB", "HML"]]
            X = sm.add_constant(X)
            y = regression_data["Excess Return"]

            X, y = X.dropna(), y.loc[X.index]
            if X.empty or y.empty:
                excluded_ciks.append({"CIK": cik, "Window": window, "Reason": "Insufficient valid data after dropna()"})
                continue

            try:
                model = sm.OLS(y, X).fit()
                if model.rsquared is None or np.isnan(model.rsquared) or np.isinf(model.rsquared):
                    raise ValueError("Invalid Regression Results (R squared is NaN or inf)")
                else:
                    results.append({
                    "CIK": cik,
                    "Window": window,
                    "Alpha": round(model.params.get("const", None), 4),
                    "Beta_Mkt-RF": round(model.params.get("Mkt-RF", None), 4) if model.params.get("Mkt-RF", None) is not None else None,
                    "Beta_SMB": round(model.params.get("SMB", None), 4) if model.params.get("SMB", None) is not None else None,
                    "Beta_HML": round(model.params.get("HML", None), 4) if model.params.get("HML", None) is not None else None,
                    "R-squared": round(model.rsquared, 4)
                })
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
                excluded_ciks.append({"CIK": cik, "Window": window, "Reason": f"Regression failure: {str(e)}"})
                continue

                
    # Save excluded CIKs to CSV
    if excluded_ciks:
        post_excluded_df = pd.DataFrame(excluded_ciks)
        post_excluded_path = os.path.join(output_folder, f"Excluded_CIKs_{year}_PostEvent.csv")
        post_excluded_df.to_csv(post_excluded_path, index=False)
        logging.info(f"Excluded CIKs saved to {post_excluded_path}")

    return pd.DataFrame(results)

def final_report(merged_data, post_merged_data, year):
    if not isinstance(merged_data, pd.DataFrame) or not isinstance(post_merged_data, pd.DataFrame):
        raise TypeError("merged_data and post_merged_data must be DataFrames")
        
        # Combine both datasets
    combined_data = pd.concat([merged_data, post_merged_data], ignore_index=True)
        
        # Remove duplicates
    combined_data = combined_data.drop_duplicates(subset=["CIK", "Filing Date"])
        
        # Handle fringe cases
    combined_data = combined_data.dropna(subset=["CIK", "Filing Date"])
        
        # Log the total number of unique CIKs
    unique_ciks = combined_data["CIK"].nunique()
    logging.info(f"Total number of unique CIKs for year {year}: {unique_ciks}")
        
        # Create the final report DataFrame
    final_report_df = combined_data[["CIK", "Filing Date"]].drop_duplicates().reset_index(drop=True)
        
    return final_report_df
