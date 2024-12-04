
import QuantLib as ql
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from collections import defaultdict
from datetime import datetime
import numpy as np

class Portfolio:
    def __init__(self, initial_investment):
        """Initialize the Portfolio."""
        self.bonds = []
        self.swaps = []
        self.initial_investment = initial_investment


        self.unrealized_returns = []
        self.realized_returns = []
        self.returns = []

        self.cash_balance = self.initial_investment
        self.cash_ledger = [{"amount": 0, "date": None, "description": f"Initializing portfolio with ${self.initial_investment:.2f} initial investment"}]

        self.current_investment = 0
        self.market_value = 0
        self.pl = 0
        self.realized_pl = 0
        self.weighted_yield = 0
        self.portfolio_yield = 0
        self.macaulay_duration = 0
        self.modified_duration = 0
        self.dollar_duration = 0
        self.dv01 = 0
        self.convexity = 0
        self.approximate_convexity = 0
        
        self.total_duration = 0
        self.total_dv01 = 0
        self.total_market_value = 0

        self.portfolio_df = pd.DataFrame()
        self.portfolio_cash_flows = None

    def update(self):
        """Calculate and update portfolio."""

        self.market_value = sum(bond.market_value for bond in self.bonds)
        self.current_investment = sum(bond.notional for bond in self.bonds)
        self.weighted_yield = sum((bond.market_value / self.market_value) * bond.yield_to_maturity for bond in self.bonds)
        self.portfolio_cash_flows = self.aggregate_cash_flows()
        self.portfolio_yield = self.calculate_portfolio_yield()
        
        self.macaulay_duration = self.calculate_portfolio_duration()
        self.modified_duration = self.macaulay_duration / (1 + self.portfolio_yield) if self.macaulay_duration > 0 else 0.00001
        self.weighted_modified_duration = sum((abs(bond.market_value / self.market_value) * bond.modified_duration) for bond in self.bonds)
        self.dollar_duration = self.modified_duration * self.market_value / 100
        self.dv01 = self.dollar_duration / 100
        self.convexity = self.calculate_portfolio_convexity()
        
        self.approximate_dv01 = self.calculate_approximate_dv01() # Reflects 1 basis point (0.01%) change
        self.approximate_convexity = self.calculate_approximate_portfolio_convexity()
        
        self.mtm = sum(bond.mtm for bond in self.bonds) + sum(swap.mtm for swap in self.swaps)
        self.accrual = sum(bond.accrual for bond in self.bonds) + sum(swap.accrual for swap in self.swaps)
        self.pl = sum(bond.pl for bond in self.bonds) + sum(swap.pl for swap in self.swaps)
        self.realized_pl = self.calculate_realized_pl()
        
        self.total_duration = self.modified_duration + sum(swap.duration for swap in self.swaps)
        self.total_dv01 = self.dv01 + sum(swap.dv01 for swap in self.swaps)
        self.total_market_value = self.market_value + sum(swap.market_value for swap in self.swaps) + self.cash_balance

# Rebalancing portfolio

    @staticmethod
    def validate_and_format_date(date):
        """
        Ensure the date is in the correct format (YYYY-MM-DD).
        Parameters:
            date (str or datetime): The date to validate and format.
        Returns:
            str: Formatted date in YYYY-MM-DD.
        """
        if isinstance(date, str):
            try:
                # Attempt to parse string into a datetime object
                date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid date format: {date}. Expected YYYY-MM-DD.")
        elif isinstance(date, datetime):
            # Already a datetime object, no conversion needed
            pass
        else:
            raise TypeError(f"Invalid date type: {type(date)}. Expected string or datetime.")
        
        return date.strftime("%Y-%m-%d")

    def add_bond(self, bond, date, price=None):
        # Validate and format the date
        date = self.validate_and_format_date(date)

        # print(bond.print_bond_info())
        self.bonds.append(bond)
        if bond.rating not in ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "NR"] and bond.notional > 25_000_000:
            print(f"Warning: Investment in bond {bond.bond_name} with rating {bond.rating} exceeds $25,000,000 limit.")
        market_value = bond.market_value
        # market_value = bond.previous_price * bond.previous_num_bonds
        self.cash_balance -= market_value
        self.cash_ledger.append(
            {"amount": -market_value, 
                "date": date, 
                "description": (f"Buying {bond.num_bonds:.2f} number of bonds {bond.cusip} with price ${market_value:.2f}" if bond.num_bonds > 0 else
                                f"Shorting {-bond.num_bonds:.2f} number of bonds {bond.cusip} with price ${-market_value:.2f}")

        })
        self.current_investment += market_value
        self.update_portfolio_df()

    def remove_bond(self, bond, date):
        """
        Remove a bond from the portfolio. Accepts either a Bond object or a CUSIP string.

        Parameters:
            bond (Bond or str): The bond object to remove or the CUSIP of the bond.
            date (str): The date of the removal transaction.
        """
        # Validate and format the date
        date = self.validate_and_format_date(date)

        # Check if `bond` is a CUSIP string
        if isinstance(bond, str):
            cusip = bond
            bond_to_remove = next((b for b in self.bonds if b.cusip == cusip), None)
            if not bond_to_remove:
                raise ValueError(f"No bond with CUSIP {cusip} found in the portfolio.")
        else:
            bond_to_remove = bond

        # Calculate proceeds from the sale (market value + accrued interest)
        sale_proceeds = bond_to_remove.market_value

        # Remove the bond
        self.bonds.remove(bond_to_remove)
        self.cash_balance += sale_proceeds
        self.cash_ledger.append({
            "amount": sale_proceeds,
            "date": date,
            "description": f"Selling {bond_to_remove.num_bonds:.2f} number of bonds {bond_to_remove.cusip} with price ${sale_proceeds:.2f}"
        })
        self.current_investment -= bond_to_remove.market_value
        self.update_portfolio_df()

    def rebalance_bond(self, bond, factor, date):
        """
        Rebalance a bond in the portfolio by adjusting its notional value.
        
        Parameters:
            bond (Bond): The bond to rebalance.
            factor (float): The percentage change in notional (e.g., 0.1 for 10% increase, -0.1 for 10% decrease).
            date (str or datetime): The date of rebalancing.
        """
        # Validate and format the date
        date = self.validate_and_format_date(date)
        num_bonds_old = bond.num_bonds

        # Calculate the cash change for the rebalancing
        cash_change = - factor * bond.market_value

        # Update the bond's notional and number of bonds
        bond.notional *= (1 + factor)
        bond.num_bonds *= (1 + factor)
        bond.update()

        # Update portfolio cash balance and ledger
        self.cash_balance += cash_change
        cash_flow_description = (
            f"Rebalancing {bond.cusip} from {num_bonds_old:.2f} to {bond.num_bonds:.2f}"
        )
        self.cash_ledger.append({
            "amount": cash_change, 
            "date": date, 
            "description": cash_flow_description
        })

        # Recalculate the current investment
        self.current_investment += cash_change
        self.update_portfolio_df()


# Updating portfolio

    def update_portfolio_df(self):
        """Update the DataFrame with current bond data."""
        data = []
        
        # Gather bond attributes and price history into rows
        for bond in self.bonds:
            bond_info = {
                "CUSIP": bond.cusip,
                "Bond Name": bond.bond_name,
                "Country": bond.country,
                "Industry Sector": bond.industry_sector,
                "Industry Group": bond.industry_group,
                "Rating": bond.rating,
                "Coupon (%)": bond.coupon,
                "Maturity": bond.maturity,
                "Frequency": bond.frequency,
                "Notional ($)": bond.notional,
                "Number of Bonds": bond.num_bonds,
                "Market Value": bond.market_value,
                "Initial Price ($)": bond.initial_price,
                "Yield to Maturity (%)": bond.yield_to_maturity,
                "Macaulay Duration": bond.macaulay_duration,
                "Modified Duration": bond.modified_duration,
                "Dollar Duration ($)": bond.dollar_duration,
                "DV01": bond.dv01,
                "Convexity": bond.convexity,
                "Approximate DV01": bond.approximate_dv01,
                "Approximate Convexity": bond.approximate_convexity,
                "Daily Accrual ($)": bond.daily_accrual,
                "Dirty Price ($)": bond.dirty_price,
                "MTM ($)": bond.mtm,
                "PL ($)": bond.pl
            }

            for date, price in bond.prices.items():
                bond_info[f"LAST_PX {date}"] = price
            
            data.append(bond_info)
        
        # Convert list of dictionaries to DataFrame
        self.portfolio_df = pd.DataFrame(data)

    def allocate_investment(self, allocation_details, investable_amount, date):
        """
        Allocate investments based on the provided allocation details.
        
        Parameters:
            allocation_details (dict): A dictionary where each key is a group name, 
                                       and each value is a dictionary with "allocation" 
                                       (percentage of investable amount) and "cusips" (list of CUSIP codes).
        """

        for group, data in allocation_details.items():
            group_investment = data["allocation"] if data["allocation"] > 1 else investable_amount * data["allocation"]
            num_bonds_in_group = len(data["cusips"])
            investment_per_bond = group_investment / num_bonds_in_group

            for bond in self.bonds:
                if bond.cusip in data["cusips"]:
                    bond.notional = investment_per_bond
                    num_bonds_old = bond.num_bonds
                    bond.num_bonds = investment_per_bond / bond.current_price
                    cash_adjustment = -investment_per_bond + bond.market_value
                    self.cash_balance += cash_adjustment
                    bond.update()
                    self.cash_ledger.append({
                        "amount": cash_adjustment, 
                        "date": date, 
                        "description": f"Rebalancing {bond.cusip} from {num_bonds_old:.2f} to {bond.num_bonds:.2f}"
                    })
        
        self.update()
        self.update_portfolio_df()

    def update_all_prices(self, prices_dict):
        """
        Update all bond prices in the portfolio and keep a history of prices.
        
        Parameters:
            prices_dict (dict): A dictionary where keys are CUSIPs and values are tuples (date, price).
        """
        for bond in self.bonds:
            if bond.cusip in prices_dict:
                date, price = prices_dict[bond.cusip]
                # Append the price to the bond's historical data
                if "prices" not in bond.__dict__:  # Ensure prices dictionary exists
                    bond.prices = {}
                bond.prices[date] = price  # Append the historical price
                # Update bond's current price
                bond.update_price(date, price)
            else:
                print(f"Price data for {bond.bond_name} not found for the given date.")
        self.update()
        self.update_portfolio_df()

# Creating statistics

    def aggregate_cash_flows(self):
        """Aggregate cash flows from all bonds in the portfolio, including each bond's payment frequency."""
        cash_flows = defaultdict(lambda: [0, 0])  # Default to [cash flow, frequency]
        
        for bond in self.bonds:
            for date, t, cf, frequency in bond.generate_cash_flows():
                cash_flows[(date, t)][0] += cf  # Aggregate cash flow amount
                cash_flows[(date, t)][1] = frequency  # Store frequency (will be the same for all entries on this date)
        
        # Sort cash flows by date and return a list of tuples (date, t, cf, frequency)
        sorted_cash_flows = sorted([(date, t, cf_freq[0], cf_freq[1]) for (date, t), cf_freq in cash_flows.items()], key=lambda x: x[0])
        
        return sorted_cash_flows

    def calculate_portfolio_yield(self):
        """Calculate the yield to maturity for the portfolio based on the aggregated cash flow schedule."""

        if not self.portfolio_cash_flows or self.market_value <= 0:
            print(f"Invalid cash flows or market value for portfolio.")
            return 0  # or fallback value

        # Define the NPV function for the portfolio with frequency adjustment
        def npv(rate):
            total_npv = sum(abs(cf) / (1 + rate) ** t for _, t, cf, freq in self.portfolio_cash_flows)
            return total_npv - abs(self.market_value)

        # Define the Brent solver
        solver = ql.Brent()

        # Attempt to solve with an initial guess and bounds for the Brent solver
        try:
            portfolio_yield = solver.solve(npv, 1e-9, 0.05, 0, 1)
        except RuntimeError as e:
            print(f"Error calculating portfolio yield with market value ${self.market_value:,.2f}")
            portfolio_yield = max(self.weighted_yield/100, 0)  # Ensure fallback is non-negative

        return portfolio_yield

    def calculate_portfolio_duration(self):
        """Calculate Macaulay duration for the portfolio based on aggregated cash flows."""
        cash_flows = self.aggregate_cash_flows()
        weights = [cf / (1 + self.portfolio_yield) ** t for _, t, cf, freq in cash_flows]
        total_weight = sum(weights)
        if total_weight == 0:
            print(f"Invalid total weight for portfolio duration. Returning 0.")
            return 0  # Return fallback duration
        durations = [w * t for (_, t, cf, freq), w in zip(cash_flows, weights)]
        return sum(durations) / total_weight

    def calculate_portfolio_convexity(self):
        """Calculate convexity for the portfolio based on aggregated cash flows."""
        portfolio_yield = self.portfolio_yield
        cash_flows = self.aggregate_cash_flows()
        weights = [cf / (1 + portfolio_yield) ** t for _, t, cf, freq in cash_flows]
        total_weight = sum(weights)
        convexities = [w * t ** 2 for (_, t, cf, freq), w in zip(cash_flows, weights)]
        return sum(convexities) / total_weight / 100 if total_weight != 0 else 0

    def npv(self, rate):
        total_npv = sum(cf / (1 + rate) ** t for _, t, cf, freq in self.portfolio_cash_flows)
        return total_npv

    def calculate_approximate_dv01(self):
        """Calculate DV01 using finite difference."""
        delta_yield = 0.0001  # 1 basis point shift

        original_yield = self.portfolio_yield
        price_up = self.npv(original_yield + delta_yield)
        price_down = self.npv(original_yield - delta_yield)
        dv01 = (price_down - price_up) / 2
        return dv01

    def calculate_approximate_portfolio_convexity(self):
        """Calculate convexity for the portfolio using a numerical method with yield shifts."""
        delta_yield = 0.0001  # 1 basis point shift

        original_yield = self.portfolio_yield
        price_up = self.npv(original_yield + delta_yield)
        price_down = self.npv(original_yield - delta_yield)
        convexity = (price_up + price_down - 2 * self.market_value) / (self.market_value * delta_yield ** 2) if self.market_value != 0 else 0
        return convexity / 100

# Hedging

    def add_swap(self, swap, current_date):
        """Add a swap to the portfolio and update risk metrics."""
        self.swaps.append(swap)
        self.cash_balance -= swap.market_value
        self.cash_ledger.append({
            "amount": -swap.market_value,
            "date": current_date,
            "description": f"Adding Swap {swap.cusip} with market value ${swap.market_value:,.2f}"
        })
        self.total_duration += swap.duration
        self.total_dv01 += swap.dv01
        self.total_market_value += swap.market_value

    def remove_swap(self, swap_cusip, current_date):
        """Remove a swap from the portfolio."""
        swap = next((s for s in self.swaps if s.cusip == swap_cusip), None)
        if swap:
            self.swaps.remove(swap)
            self.cash_balance += swap.market_value
            self.cash_ledger.append({
                "amount": swap.market_value,
                "date": current_date,
                "description": f"Adding Swap {swap.cusip} with market value ${swap.market_value:,.2f}"
            })
            self.total_market_value -= swap.market_value
            self.total_duration -= swap.duration
            self.total_dv01 -= swap.dv01

    def update_all_swaps(self, market_values):
        """Update all swaps' prices in the portfolio."""
        for swap in self.swaps:
            if swap.cusip in market_values:
                swap.update_price(self.current_date, market_values[swap.cusip])

    def sell_all(portfolio, current_date):
        """A strategy that sells all current bonds in the portfolio."""

        # Sell all existing bonds
        total_proceeds = 0
        for bond in portfolio.bonds[:]:
            total_proceeds += bond.market_value
            portfolio.remove_bond(bond, current_date)

        print(f"Sold all bonds in the portfolio. Total proceeds: ${total_proceeds:,.2f}")

# Portfolio Measurements

    def calculate_realized_pl(self):
        """
        Calculate the realized P&L based on the cash ledger.
        
        Returns:
            float: The total realized P&L.
        """
        realized_pl = 0

        for entry in self.cash_ledger:
            # Only include entries related to buying, selling, or rebalancing
            # if "Buying" in entry["description"] or "Selling" in entry["description"] or "Rebalancing" in entry["description"] or "Initializing" in entry["description"]: # or "Shorting" in entry["description"]:
            realized_pl += entry["amount"]

        return realized_pl

    def calculate_one_period_return(self, previous_total_market_value):
        """Calculate portfolio return for the current period."""
        if previous_total_market_value == 0:
            return 0  # Avoid division by zero
        return (self.total_market_value - previous_total_market_value) / previous_total_market_value

    def calculate_yield_impact(self, yield_change):
        """Calculate the impact of yield changes on the portfolio."""
        # yield_impact = 0
        # for bond in self.bonds:
        #     yield_impact += -bond.modified_duration * bond.market_value * yield_change

        yield_impact = -self.dollar_duration * yield_change
        return yield_impact

    def calculate_convexity_impact(self, yield_change):
        """Calculate the convexity impact on the portfolio's return based on yield changes."""
        # convexity_impact = 0
        # for bond in self.bonds:
        #     convexity_impact += 0.5 * bond.convexity * bond.market_value * (yield_change ** 2)
        
        convexity_impact = 0.5 * self.convexity * (yield_change ** 2) * self.market_value
        return convexity_impact

    def calculate_sharpe_ratio(self, risk_free_rate, return_history):
        """
        Calculate the Sharpe Ratio for the portfolio.

        Parameters:
            risk_free_rate (float): Annualized risk-free rate of return (e.g., 0.03 for 3%).
            return_history (dict): Dictionary of historical returns with dates as keys and returns as values.

        Returns:
            float: Annualized Sharpe Ratio.
        """
        if len(return_history) < 2:
            return None  # Not enough data to calculate standard deviation

        # Get sorted dates for the returns
        sorted_dates = sorted(return_history.keys())
        periods_per_year = 365 / ((sorted_dates[-1] - sorted_dates[0]).days / len(sorted_dates))

        # Extract returns
        returns = list(return_history.values())

        # Annualize the average return and standard deviation
        avg_return = (sum(returns) / len(returns)) * periods_per_year
        excess_return = avg_return - risk_free_rate
        std_dev = (sum((r - avg_return / periods_per_year) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
        annualized_std_dev = std_dev * (periods_per_year ** 0.5)

        # Calculate Sharpe ratio
        return round(excess_return / annualized_std_dev, 2) if annualized_std_dev != 0 else None

    def calculate_sortino_ratio(self, risk_free_rate, return_history):
        """
        Calculate the Sortino Ratio for the portfolio.

        Parameters:
            risk_free_rate (float): Annualized risk-free rate of return (e.g., 0.03 for 3%).
            return_history (dict): Dictionary of historical returns with dates as keys and returns as values.

        Returns:
            float: Annualized Sortino Ratio.
        """
        if len(return_history) < 2:
            return None

        # Extract returns
        sorted_dates = sorted(return_history.keys())
        returns = list(return_history.values())

        periods_per_year = 365 / ((sorted_dates[-1] - sorted_dates[0]).days / len(sorted_dates))

        # Annualize the average return
        avg_return = (sum(returns) / len(returns)) * periods_per_year
        excess_return = avg_return - risk_free_rate

        # Calculate downside deviation
        downside_deviation = (
            sum((min(0, r - risk_free_rate) ** 2 for r in returns)) / len(returns)
        ) ** 0.5 * (periods_per_year ** 0.5)

        return round(excess_return / downside_deviation, 2) if downside_deviation != 0 else None

    def calculate_cvar(self, return_history, confidence_level=0.05):
        """
        Calculate Conditional Value at Risk (CVaR) for the portfolio.

        Parameters:
            return_history (dict): Dictionary of historical returns with dates as keys and returns as values.
            confidence_level (float): Confidence level for CVaR (e.g., 0.05 for 95%).

        Returns:
            float: CVaR.
        """
        returns = sorted([r for r in return_history.values() if r < 0])
        var_index = int(len(returns) * confidence_level)
        return round(sum(returns[:var_index]) / var_index, 2) if var_index > 0 else None


# Printouts

    def print_portfolio_info(self):

        self.update()
        print("PORTFOLIO SUMMARY")
        print("=" * 50)
        print(f"Initial Investment: ${self.initial_investment:,.2f}")
        print(f"Currently Invested: ${self.current_investment:,.2f}")
        print(f"Cash Balance: ${self.cash_balance:,.2f}")
        print(f"Market Value: ${self.market_value:,.2f}")
        print(f"Mark-to-Market: ${self.mtm:,.2f}")
        print(f"Accrual Interest: ${self.accrual:,.2f}")
        print(f"Unrealized P&L: ${self.pl:,.2f}")
        print(f"Realized P&L: ${self.realized_pl:,.2f}")
        print(f"Growth: {self.pl/self.total_market_value*100:,.2f}%")
        print(f"Portfolio Yield: {self.portfolio_yield*100:.2f}%")
        print(f"Portfolio Yield (Weighted): {self.weighted_yield*100:.2f}%")
        print(f"Macaulay Duration: {self.macaulay_duration:.2f}")
        print(f"Modified Duration: {self.modified_duration:.2f}")
        print(f"Modified Duration (Weighted): {self.weighted_modified_duration:.2f}")
        print(f"Dollar Duration: ${self.dollar_duration:,.2f}")
        print(f"DV01: ${self.dv01:,.2f}")
        print(f"Convexity: {self.convexity:.2f}")
        print(f"Approximate DV01: ${self.approximate_dv01:,.2f}")
        print(f"Approximate Convexity: {self.approximate_convexity:.2f}")
        print(f"Number of Bonds: {len(self.bonds)}")
        print(f"Number of Swaps: {len(self.swaps)}")
        print(f"Total Duration: {self.total_duration:.2f}")
        print(f"Total DV01: ${self.total_dv01:,.2f}")
        print(f"Total Market Value: ${self.total_market_value:,.2f}")

    def export_portfolio_to_excel(self, filename="portfolio.xlsx"):
        """
        Exports a portfolio DataFrame to an Excel file with specified formatting.
        
        Parameters:
        portfolio_df (pd.DataFrame): The DataFrame containing the portfolio data.
        filename (str): The name of the Excel file to create.
        """
        # Create a writer object and specify the file name
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            # Write the DataFrame to the Excel file
            self.portfolio_df.to_excel(writer, sheet_name="Portfolio", index=False)
            
            # Access the workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets["Portfolio"]
            
            # Set the format for headers
            header_format = workbook.add_format({
                "bold": True,
                "text_wrap": True,
                "valign": "center",
                "align": "center",
                "bg_color": "#4F81BD",
                "font_color": "#FFFFFF",
                "border": 1
            })
            
            # Apply header format
            for col_num, value in enumerate(self.portfolio_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column width based on the longest value in each column
            for i, col in enumerate(self.portfolio_df.columns):
                max_width = max(self.portfolio_df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_width)
            
            # Optionally add currency format for numerical columns
            # currency_format = workbook.add_format({"num_format": "$#,##0.00"})
            # for col_num, col_name in enumerate(self.portfolio_df.columns):
            #     if pd.api.types.is_numeric_dtype(self.portfolio_df[col_name]):
            #         worksheet.set_column(col_num, col_num, 12, currency_format)
            
            print(f"Portfolio data exported successfully to {filename}")

    def print_portfolio_df(self):
        """Display the portfolio information in DataFrame format."""
        display(self.portfolio_df)

    def print_cash_ledger(self):
        for cash in self.cash_ledger:
            print(cash)

    def plot_country_distribution(self):
        """Plot the distribution of bonds by country."""
        distribution = self.portfolio_df.groupby("Country")["CUSIP"].count().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        distribution.plot(kind="bar", color="skyblue")
        plt.title("Bond Distribution by Country")
        plt.xlabel("Country")
        plt.ylabel("Number of Bonds")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_industry_sector_distribution(self):
        """Plot the distribution of bonds by industry sector."""
        distribution = self.portfolio_df.groupby("Industry Sector")["CUSIP"].count().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        distribution.plot(kind="bar", color="lightgreen")
        plt.title("Bond Distribution by Industry Sector")
        plt.xlabel("Industry Sector")
        plt.ylabel("Number of Bonds")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_industry_group_distribution(self):
        """Plot the distribution of bonds by industry group."""
        distribution = self.portfolio_df.groupby("Industry Group")["CUSIP"].count().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        distribution.plot(kind="bar", color="orange")
        plt.title("Bond Distribution by Industry Group")
        plt.xlabel("Industry Group")
        plt.ylabel("Number of Bonds")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_maturity_distribution(self, bins=None):
        """
        Plot the distribution of bonds by maturity dates.
        
        Parameters:
            bins (list, optional): A list of datetime ranges to categorize maturities.
        """
        self.portfolio_df["Maturity"] = pd.to_datetime(self.portfolio_df["Maturity"])

        if bins:
            distribution = pd.cut(self.portfolio_df["Maturity"], bins=bins).value_counts().sort_index()
            labels = [f"{bin.left:%Y-%m-%d} to {bin.right:%Y-%m-%d}" for bin in distribution.index]
        else:
            distribution = self.portfolio_df["Maturity"].dt.year.value_counts().sort_index()
            labels = distribution.index.astype(str)

        plt.figure(figsize=(10, 6))
        plt.bar(labels, distribution, color="lightcoral")
        plt.title("Bond Distribution by Maturity")
        plt.xlabel("Maturity Range")
        plt.ylabel("Number of Bonds")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def plot_risk_distribution(self):
        """Plot the distribution of bonds by risk (ratings or numeric metrics)."""
        if "Rating" not in self.portfolio_df.columns:
            print("Rating column not found in portfolio data.")
            return

        distribution = self.portfolio_df.groupby("Rating")["CUSIP"].count().sort_index()
        plt.figure(figsize=(10, 6))
        distribution.plot(kind="bar", color="purple")
        plt.title("Bond Distribution by Risk (Rating)")
        plt.xlabel("Rating")
        plt.ylabel("Number of Bonds")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_duration_distribution(self):
        """Plot the duration of bonds in the portfolio, including weighted durations based on market value and average weighted duration."""
        if "Modified Duration" not in self.portfolio_df.columns or "Market Value" not in self.portfolio_df.columns:
            print("Required columns (Modified Duration, Market Value) not found in portfolio data.")
            return

        # Calculate weighted durations
        total_market_value = self.portfolio_df["Market Value"].sum()
        self.portfolio_df["Weighted Duration"] = (
            self.portfolio_df["Modified Duration"] * self.portfolio_df["Market Value"] / total_market_value
        )

        # Calculate average weighted duration
        average_weighted_duration = self.portfolio_df["Weighted Duration"].sum()

        x_labels = self.portfolio_df["CUSIP"]

        # Create the plot
        plt.figure(figsize=(12, 6))
        bar_width = 0.4
        index = range(len(x_labels))

        # Plot individual bond durations
        plt.bar(
            index, 
            self.portfolio_df["Modified Duration"], 
            width=bar_width, 
            label="Modified Duration", 
            color="teal"
        )

        # Plot weighted durations
        plt.bar(
            [i + bar_width for i in index], 
            self.portfolio_df["Weighted Duration"], 
            width=bar_width, 
            label="Weighted Duration", 
            color="orange"
        )

        # Add a horizontal line for the average weighted duration
        plt.axhline(
            y=average_weighted_duration, 
            color="red", 
            linestyle="--", 
            linewidth=1.5, 
            label=f"Average Weighted Duration ({average_weighted_duration:.2f})"
        )

        # Add labels and legend
        plt.title("Bond Duration Distribution (Modified and Weighted)")
        plt.xlabel("CUSIP")
        plt.ylabel("Duration")
        plt.xticks([i + bar_width / 2 for i in index], x_labels, rotation=90, fontsize=8)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_long_short_yield_duration(self):
        """
        Plot the long and short positions in the portfolio by binned yield and duration.

        Creates bar charts to show the aggregate market value for long and short positions,
        grouped by binned yield to maturity and duration.
        """
        if self.portfolio_df.empty:
            print("Portfolio data is empty. Cannot create plots.")
            return

        # Filter out negative yields
        self.portfolio_df = self.portfolio_df[self.portfolio_df["Yield to Maturity (%)"] >= 0]

        # Bin the yields and durations
        yield_bins = np.arange(0, self.portfolio_df["Yield to Maturity (%)"].max() + 0.05, 0.05)
        duration_bins = np.arange(0, self.portfolio_df["Modified Duration"].max() + 2, 2)

        # Separate long and short positions
        self.portfolio_df["Position"] = self.portfolio_df["Notional ($)"].apply(lambda x: "Long" if x > 0 else "Short")
        long_positions = self.portfolio_df[self.portfolio_df["Position"] == "Long"]
        short_positions = self.portfolio_df[self.portfolio_df["Position"] == "Short"]

        # Aggregate market value by bins
        long_positions["Yield Bin"] = pd.cut(long_positions["Yield to Maturity (%)"], bins=yield_bins)
        short_positions["Yield Bin"] = pd.cut(short_positions["Yield to Maturity (%)"], bins=yield_bins)

        long_positions["Duration Bin"] = pd.cut(long_positions["Modified Duration"], bins=duration_bins)
        short_positions["Duration Bin"] = pd.cut(short_positions["Modified Duration"], bins=duration_bins)

        long_yield_binned = long_positions.groupby("Yield Bin", observed=False)["Market Value"].sum()
        short_yield_binned = short_positions.groupby("Yield Bin", observed=False)["Market Value"].sum()

        long_duration_binned = long_positions.groupby("Duration Bin", observed=False)["Market Value"].sum()
        short_duration_binned = short_positions.groupby("Duration Bin", observed=False)["Market Value"].sum()

        # Refined plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        fig.suptitle("Long and Short Positions by Yield and Duration (Binned)", fontsize=18, weight="bold")

        # Plot binned yields
        axes[0].bar(long_yield_binned.index.astype(str), long_yield_binned.values, width=0.6, color="green", label="Long")
        axes[0].bar(short_yield_binned.index.astype(str), short_yield_binned.values, width=0.6, color="red", label="Short")
        axes[0].set_title("Positions by Yield to Maturity (%) (Binned)", fontsize=14)
        axes[0].set_xlabel("Yield to Maturity (%) Bins", fontsize=12)
        axes[0].set_ylabel("Market Value ($)", fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

        # Plot binned durations
        axes[1].bar(long_duration_binned.index.astype(str), long_duration_binned.values, width=0.6, color="green", label="Long")
        axes[1].bar(short_duration_binned.index.astype(str), short_duration_binned.values, width=0.6, color="red", label="Short")
        axes[1].set_title("Positions by Modified Duration (Binned)", fontsize=14)
        axes[1].set_xlabel("Modified Duration Bins", fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()

        # Adjust layout and show
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
        plt.show()
    