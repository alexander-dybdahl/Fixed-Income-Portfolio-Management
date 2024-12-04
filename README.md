# Fixed Income Portfolio Management

## Overview

This project is designed to provide tools for managing and analyzing a fixed-income portfolio containing bonds and swaps. The main code is implemented in the **Jupyter Notebook** `fixed income portfolio management.ipynb`, which serves as the entry point for portfolio simulations, visualization, and analysis. The notebook relies on several Python modules for core calculations and functionalities.

## Project Structure

### Main Notebook
- **`fixed income portfolio management.ipynb`**:
  - Central notebook for portfolio management.
  - Implements portfolio strategies, time-step processing, and visualizations.
  - Demonstrates examples of adding bonds and swaps to the portfolio, rebalancing, and calculating returns.

### Python Modules
1. **`bond.py`**:
   - Handles bond-specific calculations such as yield, duration, and convexity.
   - Provides tools for generating cash flows and tracking price changes.

2. **`swap.py`**:
   - Manages swap instruments with fixed and floating legs.
   - Calculates metrics like market value, duration, and DV01.
   - Integrates SOFR and forward curves for floating leg rate adjustments.

3. **`portfolio.py`**:
   - Manages portfolio-level calculations and aggregation.
   - Tracks bonds, swaps, and overall portfolio metrics such as total market value and duration.

4. **`portfolio_management.py`**:
   - Contains time-step processing functions.
   - Implements methods for updating portfolio metrics and calculating returns.

5. **`yield_curve.py`**:
   - Models and interpolates yield curves using Nelson-Siegel-Svensson (NSS) and other approaches.
   - Provides tools for calculating yield changes and forward rates.

6. **`strategies.py`**:
   - Implements rebalancing and hedging strategies.
   - Contains allocation rules to adjust the portfolio dynamically.

7. **`utilities.py`**:
   - Offers utility functions for logging, data management, and date operations.

## Usage

### Setting Up
1. Clone the repository and ensure all Python files are in the same directory as the notebook.
2. Install required dependencies, including `pandas`, `numpy`, `matplotlib`, and `scipy`.

### Running the Notebook
1. Open `fixed income portfolio management.ipynb` in Jupyter Notebook or your preferred environment.
2. Follow the example workflows to:
   - Add bonds and swaps to the portfolio.
   - Update prices and calculate returns.
   - Rebalance the portfolio based on predefined strategies.

### Example Workflow
Below is a simple workflow for managing a portfolio using the notebook and Python modules:

#### Adding Instruments
```python
from portfolio import Portfolio
from bond import Bond
from swap import Swap

# Initialize Portfolio
portfolio = Portfolio(initial_investment=1_000_000)

# Add Bond
bond = Bond(
    cusip="12345", 
    maturity="2030-01-01", 
    coupon=5.0, 
    notional=100_000, 
    initial_price=100, 
    frequency=2, 
    basis=0
)
portfolio.add_bond(bond)

# Add Swap
swap = Swap(
    cusip="67890", 
    notional=500_000, 
    fixed_rate=0.03, 
    floating_rate=0.025, 
    start_date="2024-01-01", 
    maturity="2029-01-01", 
    fixed_frequency=2, 
    floating_frequency=1, 
    yield_curve=None, 
    sofr_file="sofr_data.xlsx"
)
portfolio.add_swap(swap)
```

## Rebalancing and Updating Prices

Below is an example of updating bond prices, applying a hedging strategy, and processing portfolio changes over a time step.

### Updating Prices and Rebalancing
```python
from strategies import hedging_strategy
from portfolio_management import process_time_step

# Update Prices
price_updates = {"12345": ("2024-01-02", 102)}  # Example price update for a bond
portfolio.update_all_prices(price_updates)

# Apply Hedging Strategy
metrics, portfolio, next_date = hedging_strategy(
    portfolio=portfolio, 
    yield_curve=yield_curve,
    previous_date="2024-01-01", 
    current_date="2024-01-02", 
    sheet_data=sheet_data,
    return_history=return_history,
    rebalance_strategy=hedging_strategy,
    strategy_args={"target_duration": 0,
                   "treasury_cusip": "912810RP5",
                    "initial_date": "2024-08-14"}
)
```

### Processing a Time Step

Below is an example of how to process a time step in the portfolio, including updating metrics, returns, and applying strategies.

#### Example Code
```python
from portfolio_management import process_time_step

# Example of processing a single time step
metrics, updated_portfolio, updated_date = process_time_step(
    portfolio=portfolio,
    yield_curve=None,  # Replace with a valid yield curve object
    current_date="2024-01-02",
    previous_date="2024-01-01",
    sheet_data=pd.DataFrame({  # DataFrame with updated price data
        "CUSIP": ["12345", "67890"],
        "Market_mid_price": [102, 99],
    }),
    return_history={},  # Dictionary tracking historical returns
    rebalance_strategy=None,  # Optional rebalancing function
    strategy_args={"target_duration": 5},  # Arguments for the rebalancing strategy
)

# Output updated metrics and portfolio
print(metrics)
print(updated_portfolio)
```

#### Analyzing Yield Curve and Durations

The yield curve and portfolio durations can be plotted using the `plot_curve_with_durations_and_price_change` function. This allows you to compare changes in yield curves and durations of bonds and swaps over time.

```python
# Paths to the yield curve data files for the years 2020 to 2024
files = [
    "daily-treasury-rates-2020.csv",
    "daily-treasury-rates-2021.csv",
    "daily-treasury-rates-2022.csv",
    "daily-treasury-rates-2023.csv",
    "daily-treasury-rates-2024.csv"
]

# Initialize and load the yield curve with multiple files
yield_curve = YieldCurve()
yield_curve.load_multiple_files(files)

# Plot a specific date or all dates in the range
yield_curve.plot_curve("2023-08-31")

# Plot for a date range
yield_curve.plot_all_curves(start_date="2024-08-14", end_date="2024-12-31", interval=10)

# Loading Nelson-Siegel-Svensson parameters
yield_curve.load_nss_parameters("feds200628.csv")

# Plot the yield curve for a specific date
yield_curve.plot_curve("2023-08-31")

# Plot multiple yield curves for different dates
dates = ["2024-08-14", "2024-09-18", "2024-10-31"]
yield_curve.plot_multiple_nss_curves(dates)
```

### Key Outputs

When processing a time step, the following key outputs are generated:

1. **Metrics**: A dictionary containing updated portfolio metrics, such as:
   - Portfolio market value
   - Unrealized and realized P&L
   - Portfolio duration and DV01
   - Yield changes and impacts
   - Risk-adjusted return metrics like Sharpe ratio, Sortino ratio, and CVaR

2. **Updated Portfolio**: The portfolio object after applying updates, strategies, and rebalancing.

3. **Updated Date**: The date processed in the time step.

#### Example of Accessing Metrics
```python
# Example of accessing metrics after a time step
print("Portfolio Market Value:", metrics["portfolio_market_value"])
print("Portfolio Duration:", metrics["portfolio_duration"])
print("Unrealized P&L:", metrics["unrealized_pl"])
print("Sharpe Ratio:", metrics["sharpe_ratio"])
```

### Data Visualization

The updated return history and portfolio metrics can be visualized to analyze portfolio performance over time.

#### Plotting Portfolio Returns
To visualize portfolio returns, you can use the `plot_returns` function. This will display a time series plot of portfolio returns.

```python
from utilities import plot_return_history

# Visualize returns over time
plot_return_history(return_history=return_history)
```

## Data Sources

The project uses several external data sources for analysis and calculations. Below are the details:

1. **FEDS Nominal Yield Curve Data**
   - Source: [Federal Reserve - Nominal Yield Curve](https://www.federalreserve.gov/data/nominal-yield-curve.htm)
   - File: `feds2000628.csv`
   - Description: This data provides the nominal yield curve estimates necessary for yield curve modeling and analysis.

2. **Daily Treasury Rates**
   - Source: [US Department of the Treasury - Daily Treasury Yield Curve Rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2024)
   - File: `daily-treasury-rates.csv`
   - Description: The daily treasury rates are used to validate and compare the yield curve calculations.

3. **SOFR (Secured Overnight Financing Rate)**
   - Source: [New York Federal Reserve - SOFR Rates](https://www.newyorkfed.org/markets/reference-rates/sofr)
   - File: `sofr.xlsx`
   - Description: The SOFR data is used in swap calculations, particularly for floating rate estimations in interest rate swaps.
