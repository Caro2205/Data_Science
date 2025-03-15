import pandas as pd
import os
import argparse
import logging
from utils import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_stock_files(data_folder, year):

    year_folder = os.path.join(data_folder, str(year))
    if not os.path.exists(year_folder):
        raise FileNotFoundError(f"Year folder {year_folder} not found.")


    stock_files = sorted([os.path.join(year_folder, f) for f in os.listdir(year_folder) if f.endswith(".parquet") and f.startswith(f"{year}_Stock_")])
    post_stock_files = sorted([os.path.join(year_folder, g) for g in os.listdir(year_folder) if g.endswith(".parquet") and g.startswith(f"{year}_Post_Stockprices")])

    if not stock_files or not post_stock_files:
        raise FileNotFoundError(f"No stock files found in {year_folder}.")

    stock_data_list = []
    post_stock_data_list = []
    
    
    for stock_file in stock_files:
        try:
            stock_data = pd.read_parquet(stock_file)
            logging.info(f"Loaded {stock_file}, shape: {stock_data.shape}")
            if stock_data.empty:
                logging.warning(f"Stock data {stock_file} is empty.")
                continue
            stock_data_list.append(stock_data)
        except Exception as e:
            logging.error(f"Error reading {stock_file}: {e}")
            continue

    if len(stock_data_list) == 0:
        raise ValueError("All stock data files are empty or could not be read.")
    
    
    for post_stock_file in post_stock_files:
        try:
            post_stock_data = pd.read_parquet(post_stock_file)
            logging.info(f"Loaded {post_stock_file}, shape: {post_stock_data.shape}")
            if post_stock_data.empty:
                logging.warning(f"Stock data {post_stock_file} is empty.")
                continue
            post_stock_data_list.append(post_stock_data)
        except Exception as e:
            logging.error(f"Error reading {post_stock_file}: {e}")
            continue

    if len(post_stock_data_list) == 0:
        raise ValueError("All stock data files are empty or could not be read.")

    return stock_data_list, post_stock_data_list

def compare_ciks(stock_data_list, post_stock_data_list, output_folder, year):
    pre_event_ciks = set()
    post_event_ciks = set()

    for stock_data in stock_data_list:
        if not isinstance(stock_data, pd.DataFrame):
            continue
        pre_event_ciks.update(stock_data["CIK"].unique())

    for post_stock_data in post_stock_data_list:
        if not isinstance(post_stock_data, pd.DataFrame):
            continue
        post_event_ciks.update(post_stock_data["CIK"].unique())

    missing_in_post = pre_event_ciks - post_event_ciks
    missing_in_pre = post_event_ciks - pre_event_ciks

    missing_data = {
        "Missing in Post-Event": list(missing_in_post),
        "Missing in Pre-Event": list(missing_in_pre)
    }

    missing_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in missing_data.items()]))
    missing_path = os.path.join(output_folder, f"Missing_CIKs_{year}.csv")
    missing_df.to_csv(missing_path, index=False)
    logging.info(f"Missing CIKs report saved to {missing_path}")

def generate_report(stock_data_list, post_stock_data_list, output_folder, year):

    report_data = []
    post_report_data = []
    
    for stock_data in stock_data_list:
        if not isinstance(stock_data, pd.DataFrame):
            continue
    
    for post_stock_data in post_stock_data_list:
        if not isinstance(post_stock_data, pd.DataFrame):
            continue
        
        stock_df = stock_data.copy()
        post_stock_df = post_stock_data.copy()
        
        # Count total observations per CIK
        datapoints = stock_df.groupby("RIC").size()
        post_datapoints = post_stock_df.groupby("RIC").size()
        
        # Count dropped observations (rows where 'Price Close' is missing)
        dropped = stock_df[stock_df["Price Close"].isna()].groupby("RIC").size()
        post_dropped = post_stock_df[post_stock_df["Price Close"].isna()].groupby("RIC").size()
        
        # Merge results
        report_df = pd.DataFrame({
            "RIC": datapoints.index,
            "Datapoints": datapoints.values,
            "Dropped": dropped.reindex(datapoints.index, fill_value=0).values
        })

        post_report_df = pd.DataFrame({
            "RIC": post_datapoints.index,
            "Datapoints": post_datapoints.values,
            "Dropped": post_dropped.reindex(post_datapoints.index, fill_value=0).values
        })
        
        report_data.append(report_df)
        post_report_data.append(post_report_df)
    
    if report_data and post_report_data:
        final_report = pd.concat(report_data, ignore_index=True)
        post_report = pd.concat(post_report_data, ignore_index=True)
        report_path = os.path.join(output_folder, f"Report_{year}.csv")
        post_report_path = os.path.join(output_folder, f"PostReport_{year}.csv")
        final_report.to_csv(report_path, index=False)
        post_report.to_csv(post_report_path, index=False)
        logging.info(f"Report saved to {report_path}")
    else:
        logging.warning("No data available for report generation.")

def main(data_folder, year):
    if not (2000 <= year <= 2030):
        raise ValueError("Invalid year. Please provide a year between 2000 and 2030.")

    year_folder = os.path.join(data_folder, str(year))
    fama_data_path = os.path.join(data_folder, "fama_data_daily.csv")
    cik_path = os.path.join(year_folder, f"cik_{year}.parquet")
    ric_path = os.path.join(year_folder, f"ric_{year}.parquet")
    output_folder = os.path.join(year_folder, f"Results_{year}")
    os.makedirs(output_folder, exist_ok=True)

    for file_path in [fama_data_path, cik_path, ric_path]:
        if not os.path.exists(file_path):
            logging.error(f"Required file not found: {file_path}")
            return

    try:
        stock_data_list, post_stock_data_list = load_stock_files(data_folder, year)
        logging.info("Stock data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading stock data: {e}")
        return

    try:
        fama_data, cik_data, ric_data = load_data(fama_data_path, cik_path, ric_path)
        logging.info("All data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    for i, (stock_data, post_stock_data) in enumerate(zip(stock_data_list, post_stock_data_list), start=1):
        stock_filename = f"Stock_{year}_{i}"
        post_stock_filename = f"Post_Stockprices_{year}"
        
        if not isinstance(stock_data, pd.DataFrame) or not isinstance(post_stock_data, pd.DataFrame):
            logging.error(f"Unexpected data type at index {i}: stock_data={type(stock_data)}, post_stock_data={type(post_stock_data)}")
            continue
        
        if stock_data.empty:
            logging.warning(f"Skipping empty stock dataset at index {i}.")
            continue
        if post_stock_data.empty:
            logging.warning(f"Skipping empty post-event stock dataset at index {i}.")
            continue

        try:
            merged_data, post_merged_data = merge_stock_data(stock_data, post_stock_data, ric_data)
            organized_data, post_organized_data = organize_and_calculate_returns(merged_data, post_merged_data)
            filed_data, post_filed_data = add_filing_date(organized_data, post_organized_data, cik_data)
            complete_df, post_complete_df = add_fama_french_factors(filed_data, post_filed_data, fama_data)
        except Exception as e:
            logging.error(f"Error during preprocessing for {stock_filename}: {e}")
            continue

        try:
            regression_df = perform_fama_french_regression(complete_df, output_folder, year)
        except Exception as e:
            logging.error(f"Error during pre-event regression for {stock_filename}: {e}")
            regression_df = pd.DataFrame()

        try:
            post_regression_df = perform_post_regression(post_complete_df, output_folder, year)
        except Exception as e:
            logging.error(f"Error during post-event regression for {stock_filename}: {e}")
            post_regression_df = pd.DataFrame()

        if not regression_df.empty:
            output_path = os.path.join(output_folder, f"Results_{year}_{i}.csv")
            regression_df.to_csv(output_path, index=False)
            logging.info(f"Pre-event results saved to {output_path}")
        else:
            logging.warning(f"Skipping pre-event regression results for {stock_filename} due to insufficient data.")

        if not post_regression_df.empty:
            post_output_path = os.path.join(output_folder, f"Post_Results_{year}.csv")
            post_regression_df.to_csv(post_output_path, index=False)
            logging.info(f"Post-event results saved to {post_output_path}")
        else:
            logging.warning(f"Skipping post-event regression results for {post_stock_filename} due to insufficient data.")
    
    try:
        generate_report(stock_data_list, post_stock_data_list, output_folder, year)
        logging.info("Report generation completed successfully.")
    except Exception as e:
        logging.error(f"Error generating report for {year}: {e}")

    try: 
        compare_ciks(regression_df, post_regression_df, output_folder, year)
        logging.info("Generated Missing CIKs report.")
    except Exception as e:
        logging.error(f"Error generating Missing CIKs report for {year}: {e}")

    try:
        out_path = os.path.join(output_folder, f"Final_Report_{year}.parquet")
        final = final_report(filed_data, post_filed_data, year)
        final.to_parquet(out_path)
    except Exception as e:
        logging.error(f"Error generating final report: {e}")
    
    logging.info("All files processed successfully.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data for expected returns.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the folder containing the data files')
    parser.add_argument('--year', type=int, required=True, help='Year of the data files')
    args = parser.parse_args()

    try:
        main(args.data_folder, args.year)
    except Exception as e:
        logging.error(f"Fatal error: {e}")

