import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def filter_bonds_from_excel(file_path, sheet_name, target_years, threshold_years=1, only_government=False, only_us=False):
    """
    Reads an Excel sheet and filters bonds based on target maturities calculated as offsets from a reference date.
    
    Parameters:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to read.
        target_years (list of int): Years from the reference date to calculate target maturities.
        threshold_years (int): The threshold in years for filtering around the maturities.
        only_government (bool): If True, only include government treasury bonds.
        
    Returns:
        dict: A dictionary with maturities as keys and lists of CUSIPs as values.
    """
    # Load the Excel sheet
    dataframe = pd.read_excel(file_path, sheet_name=sheet_name)

    # Ensure "MATURITY" is parsed as a datetime column
    dataframe["MATURITY"] = pd.to_datetime(dataframe["MATURITY"])

    # Reference date for calculating maturities
    reference_date = datetime(2024, 8, 14)

    # Calculate target maturities
    maturities = [reference_date + timedelta(days=365 * year) for year in target_years]

    # Prepare the results dictionary
    results = {}

    threshold_days = threshold_years * 365  # Convert years to days
    for target_maturity in maturities:
        # Calculate the range for filtering
        maturity_range = (
            target_maturity - timedelta(days=threshold_days),
            target_maturity + timedelta(days=threshold_days),
        )

        # Filter bonds by maturity range
        filtered_bonds = dataframe[
            (dataframe["MATURITY"] >= maturity_range[0]) & 
            (dataframe["MATURITY"] <= maturity_range[1])
        ]

        # Optionally filter for government bonds
        if only_government:
            filtered_bonds = filtered_bonds[filtered_bonds["Type"] == "Government"]

        if only_us:
            filtered_bonds = filtered_bonds[filtered_bonds["COUNTRY"] == "US"]

        # Collect CUSIPs for this maturity
        results[target_maturity.strftime('%Y-%m-%d')] = filtered_bonds["CUSIP"].tolist()

    return results

def plot_return_history(return_history):
    """
    Plots daily returns and cumulative returns over time based on a return history dictionary.

    Parameters:
        return_history (dict): Dictionary with dates as keys and returns as values.
    """
    # Convert return history to DataFrame
    return_df = pd.DataFrame(list(return_history.items()), columns=['Date', 'Return'])
    return_df['Date'] = pd.to_datetime(return_df['Date'])
    return_df.sort_values('Date', inplace=True)

    # Calculate cumulative return
    return_df['Cumulative Return'] = (1 + return_df['Return']).cumprod() - 1

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(return_df['Date'], return_df['Cumulative Return'], label='Cumulative Return', alpha=0.9, color='blue')
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
