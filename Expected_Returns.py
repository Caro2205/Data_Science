import pandas as pd
import os
import argparse
import logging
from datetime import datetime, timedelta
from uts import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def final_report(return_file, year):

    report_file = pd.DataFrame(return_file).copy()
    
    report_file = report_file.drop(columns=["Window", "Expected Return"], errors="ignore")

    report_file = report_file.drop_duplicates(subset=["CIK", "Filing Date"])

    report_file["Filing Date"] = pd.to_datetime(report_file["Filing Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    unique_ciks = report_file["CIK"].nunique()
    logging.info(f"Total number of unique CIKs for year {year}: {unique_ciks}")

    report_file = report_file[["CIK", "Filing Date"]].drop_duplicates().reset_index(drop=True)

    return report_file



def main(input_folder, year):

    if not (2000 <= year <= 2030):

        raise ValueError("Invalid Year. Provide a year between 2000 and 2030")
    
    #Path Processing
    year_folder = os.path.join(input_folder, str(year))
    results_folder = os.path.join(year_folder, f"Results_{year}")

    fama_path = os.path.join(input_folder, "fama_data_daily.csv")
    pre_event_data_path = os.path.join(results_folder, f"Results_{year}_1.csv")
    post_event_data_path = os.path.join(results_folder, f"Post_Results_{year}.csv")
    filing_path = os.path.join(results_folder, f"Final_Report_{year}.parquet")
    output_folder = os.path.join(year_folder, f"Results_{year}")

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"Expected_Returns_{year}.csv")

    for file_path in [fama_path, pre_event_data_path, post_event_data_path]:
        if not os.path.exists(file_path):
            logging.error(f"Required file not found: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")

    #Load Data
    try: 
        fama_df, pre_df, post_df, filing_df = load_data(fama_path, filing_path, pre_event_data_path, post_event_data_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading files: {e}.")
        return
    
    if not isinstance(pre_df, pd.DataFrame) or not isinstance(post_df, pd.DataFrame) or not isinstance(filing_df, pd.DataFrame):
        raise TypeError("Pre Event data and Post Event data must be a Dataframe.")
    
    filing_df["CIK"] = filing_df["CIK"].astype(str).str.zfill(10)
    pre_df["CIK"] = pre_df["CIK"].astype(str).str.zfill(10)
    post_df["CIK"] = post_df["CIK"].astype(str).str.zfill(10)
    
    missing_pre_event_ciks = set(filing_df["CIK"]) - set(pre_df["CIK"])
    missing_post_event_ciks = set(filing_df["CIK"]) - set(post_df["CIK"])

    logging.warning(f"Missing CIKs in pre-event regression: {len(missing_pre_event_ciks)}")
    logging.warning(f"Missing CIKs in post-event regression: {len(missing_post_event_ciks)}")

    #Merge data
    try:
        merged_pre_event, merged_post_event = add_filing_date(pre_df,post_df,filing_df)
        logging.info("Filing Date added successfully.")
    except Exception as e:
        logging.error(f"Error Adding Filing Date: {e}.")
        return

    if not isinstance(merged_pre_event, pd.DataFrame) or not isinstance(merged_post_event, pd.DataFrame):
        raise TypeError("Pre Event data and Post Event data must be a Dataframe.")
    if not isinstance(fama_df, pd.DataFrame):
        raise TypeError("Fama Data must be a Dataframe")
    
    #Logging Unique CIKs
    total_unique_ciks = len(merged_pre_event["CIK"].unique())
    logging.info(f"Total unique CIKs for {year}: {total_unique_ciks}")

    #Check Columns
    required_cols = ["Filing Date", "CIK", "Window", "Alpha", "Beta_Mkt-RF", "Beta_SMB", "Beta_HML", "R-squared"]

    for df, name in zip([merged_pre_event, merged_post_event], ["Pre_Event", "Post_Event"]):
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {name} file: {', '.join(missing_cols)}")

    #Merge Data for easy handling   
    merged_df = pd.concat([merged_pre_event,merged_post_event])

    if merged_df["CIK"].isna().sum() > 0:
        logging.error(f"merged_df contains {merged_df['CIK'].isna().sum()} NaN values before expected return calculations.")

    logging.info("Final unique CIK count after merging: " + str(merged_df["CIK"].nunique()))

    logging.info(f"Unique Window values before sorting: {merged_df['Window'].unique()}")

    #Ordering windows
    window_order = {250: 1, 30: 2, 60: 3, 90: 4, "1_day_before_after": 5}
    merged_df["Window"] = merged_df["Window"].astype(str)
    merged_df["Window_Order"] = merged_df["Window"].map(window_order)
    merged_df = merged_df.sort_values(by=["CIK", "Window_Order"]).drop(columns=["Window_Order"])

    #Ensure DateTime
    fama_df["Date"] = pd.to_datetime(fama_df["Date"])
    merged_df["Filing Date"] = pd.to_datetime(merged_df["Filing Date"])

    #Expected Return Calculation
    expected_return_list = []

    for _, row in merged_df.iterrows():
        cik = row["CIK"]

        if pd.isna(cik):
            logging.error(f"NaN found in CIK inside loop! Row Data: {row}")

        alpha = row["Alpha"]
        beta_mkt = row["Beta_Mkt-RF"]
        beta_smb = row["Beta_SMB"]
        beta_hml = row["Beta_HML"]
        filing_date = row["Filing Date"]
        window = row["Window"]

        # Ensure no NaN values affect calculations
        if pd.isna(alpha) or pd.isna(beta_mkt) or pd.isna(beta_smb) or pd.isna(beta_hml):
            logging.error(f"NaN detected in regression values! CIK: {cik}, Alpha: {alpha}, Beta_Mkt-RF: {beta_mkt}, Beta_SMB: {beta_smb}, Beta_HML: {beta_hml}")


        if window == "250": #pre-event
            start_date = filing_date - pd.Timedelta(days=250)
            end_date = filing_date
        elif window == "1_day_before_after":  # Special case
            start_date = filing_date - pd.Timedelta(days=1)
            end_date = filing_date + pd.Timedelta(days=1)
        else:
            start_date = filing_date
            end_date = filing_date + pd.Timedelta(days=int(window))

        factor_window = pd.DataFrame(fama_df[(fama_df["Date"] >= start_date) & (fama_df["Date"] < end_date)])

        if factor_window.empty:
            continue

        avg_mkt_rf = factor_window["Mkt-RF"].mean()
        avg_smb = factor_window["SMB"].mean()
        avg_hml = factor_window["HML"].mean()

        expected_return = alpha + (beta_mkt * avg_mkt_rf) + (beta_smb * avg_smb) + (beta_hml * avg_hml)

        expected_return_list.append({
            "Filing Date": filing_date,
            "CIK": cik,
            "Window": window,
            "Expected Return": round(expected_return, 4)
        }) 

    expected_df = pd.DataFrame(expected_return_list)
    expected_df.to_csv(output_path, index=False)
    logging.info(f"Expected Returns saved to {output_path}")

    try:
        report_path = os.path.join(output_folder, f"CIK_Report_{year}.parquet")
        report_df = final_report(expected_df, year)
        report_df.to_parquet(report_path)
        logging.info(f"Generated Final Report for year: {year}")
    except Exception as e:
        logging.error(f"Error generating final report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Expected Returns")
    parser.add_argument('--data_folder', type=str,required=True, help="Path to the data folder")
    parser.add_argument('--year', type=int, required=True, help="Year of the filed data")
    args = parser.parse_args()

    try:
        main(args.data_folder, args.year)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        



    




    



