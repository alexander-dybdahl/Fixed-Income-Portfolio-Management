import matplotlib.colors as mcolors
from datetime import datetime
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

class YieldCurve:
    def __init__(self, data_file=None):
        self.yield_data = pd.DataFrame()
        self.nss_params = pd.DataFrame()
        self.maturities = {
            "1 Mo": 1 / 12,
            "2 Mo": 2 / 12,
            "3 Mo": 3 / 12,
            "4 Mo": 4 / 12,
            "6 Mo": 6 / 12,
            "1 Yr": 1,
            "2 Yr": 2,
            "3 Yr": 3,
            "5 Yr": 5,
            "7 Yr": 7,
            "10 Yr": 10,
            "20 Yr": 20,
            "30 Yr": 30
        }
        self.sorted_maturities = sorted((value, key) for key, value in self.maturities.items())
        self.maturity_years, self.maturity_labels = zip(*self.sorted_maturities)
        self.maturity_labels = list(self.maturity_labels)
        
        if data_file:
            self.load_data(data_file)

    def load_data(self, data_file):
        data = pd.read_csv(data_file, parse_dates=["Date"])
        data.columns = data.columns.str.strip()
        self.yield_data = pd.concat([self.yield_data, data], ignore_index=True)
        self.yield_data.drop_duplicates(subset="Date", inplace=True)
        self.yield_data.sort_values(by="Date", inplace=True)

    def load_multiple_files(self, files):
        for file in files:
            self.load_data(file)

    def load_nss_parameters(self, data_file):
        """
        Load Nelson-Siegel-Svensson parameters from a CSV file and preprocess for efficient lookup.

        Parameters:
            data_file (str): Path to the CSV file containing NSS parameters.
        """
        # Load the NSS parameter data
        self.nss_params = pd.read_csv(data_file, parse_dates=["Date"])
        self.nss_params.columns = self.nss_params.columns.str.strip()

        # Remove rows with any NA in the NSS columns
        nss_columns = ['BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']
        self.nss_params.dropna(subset=nss_columns, how='any', inplace=True)

        # Preprocess: Create a dictionary mapping dates to parameter tuples
        self.nss_lookup = {
            row.Date: (row.BETA0, row.BETA1, row.BETA2, row.BETA3, row.TAU1, row.TAU2)
            for row in self.nss_params.itertuples()
        }

        # Store the sorted dates for interpolation
        self.nss_dates = sorted(self.nss_lookup.keys())

    def get_interpolated_rate(self, current_date, target_maturity):
        """Retrieve or interpolate yield data for a given date and maturity."""
        current_date = pd.to_datetime(current_date).normalize()
        if current_date not in self.yield_data['Date'].values:
            # Use the most recent prior date
            closest_date = self.yield_data[self.yield_data['Date'] <= current_date]['Date'].max()
            if pd.isna(closest_date):
                raise ValueError(f"No data available before {current_date}")
            print(f"Using closest available date: {closest_date} for {current_date}")
            current_date = closest_date

        row = self.yield_data.loc[self.yield_data['Date'] == current_date]
        rates = row[self.maturity_labels].values.flatten()
        spline = CubicSpline(self.maturity_years, rates)
        return spline(target_maturity) / 100

    def get_nss_yield(self, date, maturity):
        """
        Calculate the yield using the NSS model for a given date and maturity.

        Parameters:
            date (datetime or str): Date for which to calculate the yield.
            maturity (float): Maturity in years.

        Returns:
            float: Yield for the given maturity.
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)

        # Find the closest date if the exact date is not available
        if date not in self.nss_lookup:
            # Get the most recent date before the requested date
            closest_date = max(d for d in self.nss_dates if d <= date)
            if closest_date is None:
                raise ValueError(f"No NSS parameters available for date {date}")
            date = closest_date

        # Retrieve the parameters for the closest date
        beta0, beta1, beta2, beta3, tau1, tau2 = self.nss_lookup[date]

        # Handle the case where maturity is zero
        if maturity == 0:
            return beta0 / 100  # Return the immediate short-term yield as β0 (converted to decimal)

        # Calculate the NSS yield
        term = maturity
        y_t = (
            beta0
            + beta1 * (1 - np.exp(-term / tau1)) / (term / tau1)
            + beta2 * ((1 - np.exp(-term / tau1)) / (term / tau1) - np.exp(-term / tau1))
            + beta3 * ((1 - np.exp(-term / tau2)) / (term / tau2) - np.exp(-term / tau2))
        )
        return y_t / 100  # Convert to decimal form
    
    def get_treasury_yields(self, date):
        """
        Get the treasury yields from files.
        """
        closest_date = self.yield_data.loc[self.yield_data['Date'] <= date, 'Date'].max()
        row = self.yield_data.loc[self.yield_data['Date'] == closest_date]
        rates = row[self.maturity_labels].values.flatten()
        return rates

    def calculate_yield_change(self, date1, date2, maturity=None):
        """
        Calculate the change in yields between two dates for a specific maturity or the entire curve.

        Parameters:
            date1 (datetime): The earlier date.
            date2 (datetime): The later date.
            maturity (float, optional): The specific maturity (in years) to calculate yield change. 
                                        If None, calculates average yield change across the curve.

        Returns:
            float: Yield change (in decimal format, e.g., 0.001 for 0.1%).
        """
        # Get yields for the two dates
        yields1 = self.get_treasury_yields(date1)
        yields2 = self.get_treasury_yields(date2)

        # If a specific maturity is provided, interpolate yield for that maturity
        if maturity is not None:
            y1 = self.get_nss_yield(date1, maturity) * 100
            y2 = self.get_nss_yield(date2, maturity) * 100
            return y2 - y1

        # Otherwise, calculate average yield change across the curve
        yield_changes = [(y2 - y1) for y1, y2 in zip(yields1, yields2)]

        if yield_changes:
            return sum(yield_changes) / len(yield_changes)
        else:
            print(f"No yield changes calculated for dates: {date1}, {date2}")
            return None  # or return 0 if you prefer a default value

    def plot_curve(self, dates, use_nss=False):
        """
        Plot yield curves for given dates, allowing both US Treasury Par Yield curves
        and NSS curves to be plotted together.

        Parameters:
        - dates: A single date or a list of dates to plot.
        - use_nss: If True, include NSS curves in the plot.
        """
        if isinstance(dates, str) or isinstance(dates, datetime):
            dates = [dates]

        # Convert date strings to datetime
        dates = [pd.to_datetime(date).normalize() for date in dates]

        # Set up color map for consistent colors per date
        cmap = plt.cm.get_cmap('tab10', len(dates))
        colors = {date: cmap(idx) for idx, date in enumerate(dates)}

        plt.figure(figsize=(10, 6))
        
        for date in dates:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            # Plot US Treasury Par Yield Curve
            closest_date = self.yield_data.loc[self.yield_data['Date'] <= date, 'Date'].max()
            if not pd.isna(closest_date):
                treasury_yields = self.get_treasury_yields(closest_date)
                plt.plot(self.maturity_years, treasury_yields, label=f"US Treasury ({closest_date})", 
                         linestyle='-', marker='o', color=colors[date])

            # Plot NSS Curve if available
            if use_nss:
                # Find the closest valid NSS date with non-NA values
                closest_nss_date = self.nss_params[
                    (self.nss_params['Date'] <= date) & self.nss_params.drop(columns="Date").notna().all(axis=1)
                ]['Date'].max()

                if not pd.isna(closest_nss_date):
                    maturities = np.linspace(0.1, 30, 300)
                    nss_yields = [self.get_nss_yield(closest_nss_date, m) * 100 for m in maturities]  # Convert to percentage
                    plt.plot(maturities, nss_yields, linestyle='--', label=f"NSS ({closest_nss_date})",
                            color=colors[date])
                else:
                    print(f"Warning: No valid NSS data available for {date} or earlier.")

        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield (%)")
        plt.title("Yield Curves (US Treasury vs NSS)")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.show()

    def plot_curve_with_durations(self, dates, portfolio, use_nss=True):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(dates)))

        for i, date in enumerate(dates):
            # Plot US Treasury Par Yield Curve
            closest_date = self.yield_data.loc[self.yield_data['Date'] <= date, 'Date'].max()
            treasury_yields = self.get_treasury_yields(closest_date)
            plt.plot(self.maturity_years, treasury_yields, marker='o', label=f"US Treasury ({date})", color=colors[i])

            if use_nss:
                nss_yields = [self.get_nss_yield(date, m) * 100 for m in self.maturity_years]
                plt.plot(self.maturity_years, nss_yields, linestyle='--', label=f"NSS ({date})", color=colors[i])

        # Overlay bond durations
        for bond in portfolio.bonds:
            bond_duration = bond.modified_duration
            plt.axvline(x=bond_duration, linestyle=':', alpha=0.7) #, label=f"Bond ({bond.cusip}) Duration")

        # Add portfolio duration
        portfolio_duration = portfolio.total_duration
        plt.axvline(x=portfolio_duration, linestyle='-', alpha=0.9, label=f"Portfolio Duration ({date})")

        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield (%)")
        plt.title("Yield Curves with Bond Durations")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def plot_curve_with_durations_and_price_change(self, dates, portfolio, use_nss=True):
        fig, ax1 = plt.subplots(figsize=(12, 8))
        colors = plt.cm.Set2(np.linspace(0, 1, len(dates)))

        # Plot Yield Curves
        for i, date in enumerate(dates):
            formatted_date = date.strftime("%Y-%m-%d")

            # Plot US Treasury Par Yield Curve
            closest_date = self.yield_data.loc[self.yield_data['Date'] <= date, 'Date'].max()
            treasury_yields = self.get_treasury_yields(closest_date)
            ax1.plot(self.maturity_years, treasury_yields, marker='o', label=f"US Treasury ({formatted_date})", color=colors[-i-1])

            # Plot NSS curve if enabled
            if use_nss:
                years = np.linspace(self.maturity_years[0], self.maturity_years[-1], 100)
                nss_yields = [self.get_nss_yield(date, m) * 100 for m in years]
                ax1.plot(years, nss_yields, linestyle='--', label=f"NSS ({formatted_date})", color=colors[-i-1])

        ax1.set_xlabel("Maturity (Years)")
        ax1.set_ylabel("Yield (%)")
        ax1.set_title("Yield Curves with Bond Durations and Price Changes")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        # Overlay bond durations and price change bars directly on the primary axis
        bar_positions = []
        bar_heights = []
        bond_price_changes = []
        bar_baselines = []
        bar_colors = []

        max_price_change = max(abs(bond.mtm) for bond in portfolio.bonds)

        for bond in portfolio.bonds:
            bond_duration = abs(bond.modified_duration)
            
            if bond_duration < 0:
                continue

            bond_price_change = bond.mtm  # Use MTM (price change)
            bond_baseline = self.get_nss_yield(dates[1], bond_duration) * 100 if use_nss else 0

            # Normalize the bar heights to be proportional to the largest price change
            normalized_height = (bond_price_change / max_price_change) * (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.2  # Scale to 20% of the plot height

            # Determine bar color based on price change
            bar_colors.append('green' if bond_price_change > 0 else 'red')

            # Store bar data
            bar_positions.append(bond_duration)
            bar_heights.append(normalized_height)
            bond_price_changes.append(bond_price_change)
            bar_baselines.append(bond_baseline)
            
            plt.axvline(x=bond_duration, linestyle=':', alpha=0.7, color=('lightblue' if bond.notional > 0 else 'lightcoral'))

        # Handle Swaps in the Portfolio
        max_swap_price_change = max(abs(swap.market_value) for swap in portfolio.swaps) if portfolio.swaps else 0

        for swap in portfolio.swaps:
            swap_duration = abs(swap.duration)
            swap_price_change = swap.mtm  # Use market value as P&L for swaps
            swap_baseline = self.get_nss_yield(dates[1], swap_duration) * 100 if use_nss else 0

            # Normalize the bar heights for swaps
            normalized_swap_height = (swap_price_change / max(max_price_change, max_swap_price_change)) * \
                                    (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.2  # Scale to 20% of the plot height

            # Determine bar color based on swap P&L
            bar_colors.append('green' if swap_price_change > 0 else 'red')

            # Plot swap duration as a vertical line
            plt.axvline(x=swap_duration, linestyle=':', alpha=0.7, color='purple', label=f"Swap Duration ({swap.cusip})")

            # Store swap data for the bar plot
            bar_positions.append(swap_duration)
            bar_heights.append(normalized_swap_height)
            bond_price_changes.append(swap_price_change)
            bar_baselines.append(swap_baseline)

        # Add portfolio duration
        portfolio_duration = portfolio.total_duration
        plt.axvline(x=portfolio_duration, linestyle='-', alpha=0.9, label=f"Portfolio Duration ({round(portfolio_duration, 2)})", color='lightblue')

        # Plot price change as bars at bond maturities
        bar_width = 0.4
        bars = ax1.bar(
            bar_positions,
            bar_heights,
            width=bar_width,
            bottom=bar_baselines,
            color=bar_colors,
            alpha=0.7,
            label="Portfolio Price Change (P&L)",
        )

        # Adjust annotation positions to prevent overlap
        used_positions = []

        def find_non_overlapping_position(x, y, used_positions, offset=0.10):
            for existing_x, existing_y in used_positions:
                if abs(x - existing_x) < 0.50 and abs(y - existing_y) < 0.10:
                    y += offset
            used_positions.append((x, y))
            return x, y

        # Annotate bars with values
        for bar, raw_height, price_change, bond in zip(bars, bar_heights, bond_price_changes, portfolio.bonds):
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_y() + height + (0.02 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]))
            x_pos, y_pos = find_non_overlapping_position(x_pos, y_pos, used_positions)

            ax1.text(
                x_pos,
                y_pos,
                f"${price_change:,.2f}",
                ha='center',
                fontsize=8,
            )

        # Annotate swaps with values
        for swap, raw_height, price_change in zip(portfolio.swaps, bar_heights[len(portfolio.bonds):], bond_price_changes[len(portfolio.bonds):]):
            x_pos = swap.duration
            y_pos = swap_baseline + raw_height + (0.02 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]))
            x_pos, y_pos = find_non_overlapping_position(x_pos, y_pos, used_positions)

            ax1.text(
                x_pos,
                y_pos,
                f"${price_change:,.2f}",
                ha='center',
                fontsize=8,
            )
        
        # Plot price change for portfolio maturities
        portfolio_baseline = self.get_nss_yield(dates[1], portfolio_duration) * 100 if use_nss else 0
        bars = ax1.bar(
            portfolio_duration,
            sum(bar_heights),
            width=bar_width,
            bottom=portfolio_baseline,
            color='green' if sum(bond_price_changes) > 0 else 'red',
            alpha=0.7,
            label="Bond Price Change (P&L)",
        )

        # Annotate portfolio bar with values
        x_pos, y_pos = portfolio_duration + bar_width / 2, portfolio_baseline + sum(bar_heights) + (0.02 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]))
        x_pos, y_pos = find_non_overlapping_position(x_pos, y_pos, used_positions)

        ax1.text(
            x_pos,
            y_pos,
            f"${sum(bond_price_changes):,.2f}",
            ha='center',
            fontsize=10,
            weight='bold'
        )

        ax1.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def plot_all_curves(self, start_date=None, end_date=None, interval=30):
        # Filter data for the specified date range
        data = self.yield_data
        if start_date:
            data = data[data['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['Date'] <= pd.to_datetime(end_date)]

        # Handle cases with no valid data
        if data.empty:
            raise ValueError("No data available for the specified date range.")

        # Set up color gradient from light to dark
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=0, vmax=len(data['Date'].iloc[::interval]) - 1)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        selected_dates = data['Date'].iloc[::interval]

        # Plot each yield curve with a color gradient
        for idx, date in enumerate(selected_dates):
            row = data.loc[data['Date'] == date]
            rates = row[self.maturity_labels].values.flatten()
            color = cmap(norm(idx))
            linewidth = 1.5 if idx == len(selected_dates) - 1 else 1
            ax.plot(self.maturity_years, rates, marker='o', color=color, linewidth=linewidth,
                    label=date.strftime('%Y-%m-%d') if idx % (len(selected_dates) // 5) == 0 or idx == len(selected_dates) - 1 else None)

        # Set plot labels and titles
        ax.set_xlabel("Maturity (Years)")
        ax.set_ylabel("Yield (%)")
        ax.set_title("Yield Curves Over Time")
        ax.legend(title="Date", loc='upper right', fontsize='small', ncol=2, frameon=False)

        # Apply tight layout for better spacing and add grid
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.show()
    
    def calculate_forward_rate(curve, date, t1, t2, use_nss=False):
        """
        Calculate the implied forward rate between two maturities t1 and t2.
        
        Parameters:
            curve: Instance of the YieldCurve class.
            date: The date for which to calculate the forward rate.
            t1: Start maturity (in years).
            t2: End maturity (in years).
            use_nss: If True, use the NSS curve for yields; otherwise, use interpolated Treasury rates.
            
        Returns:
            Forward rate between t1 and t2 (as a decimal).
        """
        if use_nss:
            r1 = curve.get_nss_yield(date, t1)
            r2 = curve.get_nss_yield(date, t2)
        else:
            r1 = curve.get_interpolated_rate(date, t1)
            r2 = curve.get_interpolated_rate(date, t2)
        
        # Compute discount factors
        df1 = np.exp(-r1 * t1)
        df2 = np.exp(-r2 * t2)
        
        # Compute forward rate
        forward_rate = (np.log(df1) - np.log(df2)) / (t2 - t1)
        return forward_rate

    def get_nss_discount_factor(self, date, maturity):
        """Calculate the discount factor using NSS model yields."""
        yield_rate = self.get_nss_yield(date, maturity)
        return np.exp(-yield_rate * maturity)

    def plot_multiple_nss_curves(self, dates):
        """Plot NSS yield curves for multiple dates."""
        plt.figure(figsize=(10, 6))
        for date in dates:
            if isinstance(date, str):  # Convert strings to datetime objects
                date = datetime.strptime(date, "%Y-%m-%d")
            row = self.nss_params[self.nss_params['Date'] == date]
            if not row.empty:
                maturities = np.linspace(0.1, 30, 300)
                yields = [self.get_nss_yield(date, m) for m in maturities]
                plt.plot(maturities, yields, label=date.strftime('%Y-%m-%d'))
        
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield (%)")
        plt.title("Nelson-Siegel-Svensson Yield Curves")
        plt.grid(True)
        plt.legend()
        plt.show()