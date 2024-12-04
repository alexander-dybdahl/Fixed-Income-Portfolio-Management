from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import interp1d

class Swap:
    def __init__(self, cusip, notional, fixed_rate, floating_rate, 
                 curve_date, valuation_date, start_date, maturity, 
                 fixed_frequency, floating_frequency, yield_curve, sofr_file,
                 fixed_leg_basis=0, floating_leg_basis=1, swap_type=None):
        """
        Initialize the Swap with a similar structure as the Bond class.

        Parameters:
            cusip (str): Unique identifier for the swap.
            notional (float): Notional amount of the swap.
            fixed_rate (float): Fixed interest rate of the swap (e.g., 0.025 for 2.5%).
            floating_rate (float): Initial floating rate for the swap (this will be updated periodically).
            start_date (str): Start date of the swap in 'YYYY-MM-DD' format.
            maturity (str): Maturity date of the swap in 'YYYY-MM-DD' format.
            frequency (int): Payment frequency (e.g., 2 for semiannual).
            fixed_leg_basis (int): Day count convention for the fixed leg (0 for 30/360, 1 for actual/365).
            floating_leg_basis (int): Day count convention for the floating leg (0 for 30/360, 1 for actual/365).
        """
        self.cusip = cusip
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.floating_rate = floating_rate
        self.curve_date = datetime.strptime(curve_date, "%Y-%m-%d")
        self.valuation_date = datetime.strptime(valuation_date, "%Y-%m-%d")
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.maturity = datetime.strptime(maturity, "%Y-%m-%d")
        self.fixed_frequency = fixed_frequency
        self.floating_frequency = floating_frequency
        self.fixed_leg_basis = fixed_leg_basis
        self.floating_leg_basis = floating_leg_basis
        self.swap_type = swap_type
        self.position_type = 1 if swap_type.lower=="reciever" else -1

        self.previous_market_value = 0.0
        self.market_value = 0.0

        self.load_sofr_data(sofr_file=sofr_file)
        self.yield_curve = yield_curve

        # Initialize calculations
        self.update()

    def update(self, new_curve_date=None, new_valuation_date=None):
        # Update curve and valuation dates if provided
        if new_curve_date:
            self.curve_date = new_curve_date
        if new_valuation_date:
            self.valuation_date = new_valuation_date

        self.fixed_leg_cash_flows = self.generate_fixed_leg_cash_flows()
        self.fixed_leg_princpal = self.calculate_npv(self.fixed_leg_cash_flows)
        self.fixed_leg_value = self.fixed_leg_princpal - self.accrual_fixed

        self.floating_leg_cash_flows = self.generate_floating_leg_cash_flows()
        self.floating_leg_principal = self.calculate_npv(self.floating_leg_cash_flows)
        self.floating_leg_value = self.floating_leg_principal - self.accrual_floating

        self.previous_market_value = self.market_value
        self.market_value = self.fixed_leg_value + self.floating_leg_value

        # Calculate duration, DV01, and accrual interest
        self.fixed_duration = self.calculate_macaulay_duration(self.fixed_leg_cash_flows) / (1 + self.fixed_rate)
        self.floating_duration = self.calculate_macaulay_duration(self.floating_leg_cash_flows) / (1 + self.floating_rate)
        self.duration = self.fixed_duration + self.floating_duration

        # Adjust duration for short swaps
        if self.notional < 0:
            self.duration *= -1

        # Calculate DV01
        self.dv01_fixed = self.calculate_dv01(self.fixed_leg_cash_flows, self.fixed_leg_basis)
        self.dv01_floating = self.calculate_dv01(self.floating_leg_cash_flows, self.floating_leg_basis)
        self.dv01 = self.dv01_fixed + self.dv01_floating

        self.convexity = self.calculate_convexity(self.fixed_leg_cash_flows)
        
        self.accrual = self.accrual_fixed + self.accrual_floating
    
        # Update market value, duration, etc.
        self.mtm = self.market_value - self.previous_market_value
        self.pl = self.mtm + self.accrual

        # Update duration and convexity
        self.modified_duration = self.fixed_duration + self.floating_duration
        self.dv01 = self.dv01_fixed + self.dv01_floating
        self.convexity = self.calculate_convexity(self.fixed_leg_cash_flows + self.floating_leg_cash_flows)

    def calculate_day_difference(self, start_date, end_date, basis):
        """Calculate day difference based on day count convention (basis)."""
        if basis == 0:  # 30/360 day count convention
            start_day = min(start_date.day, 30)
            end_day = min(end_date.day, 30)
            day_diff = ((end_date.year - start_date.year) * 360 + (end_date.month - start_date.month) * 30 + (end_day - start_day))
        elif basis == 1:  # Actual/365 day count convention
            day_diff = (end_date - start_date).days
        elif basis == 2:  # Actual/360 day count convention
            day_diff = (end_date - start_date).days
        elif basis == 3:  # Actual/Actual
            day_diff = (end_date - start_date).days
        elif basis == 4:  # 30E/360
            start_day = min(start_date.day, 30)
            end_day = min(end_date.day, 30)
            day_diff = (end_date.year - start_date.year) * 360 + (end_date.month - start_date.month) * 30 + (end_day - start_day)
        else:
            raise ValueError("Unsupported basis. Use 0 for 30/360, 1 for Actual/365, 2 for Actual/360, 3 for Actual/Actual, or 4 for 30E/360.")
        return day_diff

    def calculate_day_fraction(self, start_date, end_date, basis):
        """Calculate the day fraction based on day count convention."""
        days_in_year = 365 if basis in [1, 3] else 360
        day_diff = self.calculate_day_difference(start_date, end_date, basis)
        return day_diff / days_in_year

    def update_floating_rate(self, last_date, cash_flow_date):
        """
        Update the floating rate using forward rates or fallback to the SOFR rate if the forward rate cannot be calculated.
        """
        try:
            # Calculate the time fraction between the current date and the last reset date
            time_to_start = self.calculate_day_fraction(last_date, cash_flow_date, self.floating_leg_basis)
            
            # If the current date is past the last date (i.e., after the previous reset), or it's the first cash flow
            if time_to_start <= 0:
                # Use the SOFR rate for the initial cash flow or when current date exceeds available forward rate data
                self.floating_rate = self.get_sofr_rate(last_date)
            else:
                # Otherwise, calculate the forward rate for subsequent periods
                forward_rate = self.get_forward_rate(last_date, cash_flow_date)
                self.floating_rate = forward_rate
        
        except ValueError as e:
            print(f"Error calculating forward rate: {e}. Using fallback SOFR rate.")
            self.floating_rate = self.get_sofr_rate(last_date)

    def get_forward_rate(self, start_date, end_date):
        """Retrieve or interpolate the forward rate for a given time to maturity."""
        # Interpolate the zero-coupon rates for the start and end of the period
        
        time_to_start = self.calculate_day_fraction(self.curve_date, start_date, self.floating_leg_basis)
        time_to_end = self.calculate_day_fraction(self.curve_date, end_date, self.floating_leg_basis)
    
        spot_rate_start = self.yield_curve.get_nss_discount_factor(self.curve_date, time_to_start)
        spot_rate_end = self.yield_curve.get_nss_discount_factor(self.curve_date, time_to_end)
        
        # Calculate the forward rate using the formula:
        forward_rate = 1 / (time_to_end - time_to_start) * (spot_rate_start / spot_rate_end - 1)
        return forward_rate

    def load_sofr_data(self, sofr_file):
        """Load SOFR rates from an Excel file."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sofr_data = pd.read_excel(sofr_file, parse_dates=["Effective Date"], engine="openpyxl")
            self.sofr_data.columns = self.sofr_data.columns.str.strip()
            self.sofr_data.sort_values(by="Effective Date", inplace=True)
        except Exception as e:
            print(f"Error loading SOFR data: {e}")
            self.sofr_data = None

    def get_sofr_rate(self, date):
        """Fetch the most recent SOFR rate for the given date."""
        row = self.sofr_data[self.sofr_data['Effective Date'] <= date].iloc[-1]
        return row['Rate (%)'] / 100  # Convert to decimal

    def load_forward_curve(self, file_path):
        """Load the Pensford Forward Curve for the floating leg."""

        # Load the Pensford Forward Curve Excel file
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            forward_curve_data = pd.ExcelFile(file_path)
            forward_curve_raw = forward_curve_data.parse('Forward Curve', skiprows=4, header=None)

        # Extract relevant columns for Reset Date and Market Expectations
        forward_curve_cleaned = forward_curve_raw[[6, 7]]
        forward_curve_cleaned.columns = ['Reset Date', 'Market Expectations']

        # Drop rows with missing data
        forward_curve_cleaned = forward_curve_cleaned.dropna()

        # Convert Reset Date to datetime and Market Expectations to float
        forward_curve_cleaned['Reset Date'] = pd.to_datetime(forward_curve_cleaned['Reset Date'], errors='coerce')
        forward_curve_cleaned['Market Expectations'] = pd.to_numeric(forward_curve_cleaned['Market Expectations'], errors='coerce')

        # Display the cleaned forward curve data
        forward_curve_cleaned.head()

        self.forward_curve = forward_curve_cleaned

    def get_floating_forward_rate(self, reset_date):
        """Interpolate forward rate for a given reset date."""
        if not hasattr(self, 'forward_curve') or self.forward_curve.empty:
            raise ValueError("Forward curve not loaded.")
        
        forward_curve = self.forward_curve.sort_values('Reset Date')
        interp = interp1d(
            pd.to_numeric(forward_curve['Reset Date']),
            forward_curve['Market Expectations'],
            kind='linear',
            fill_value="extrapolate",
        )
        return interp(pd.to_numeric(reset_date))

    def generate_fixed_leg_cash_flows(self):
        cash_flows = []

        time_fraction = np.abs(self.calculate_day_fraction(self.valuation_date, self.start_date, self.fixed_leg_basis))
        self.accrual_fixed = self.position_type * self.notional * self.fixed_rate * time_fraction

        last_date = self.start_date
        cash_flow_date = self.increment_date(self.start_date, self.fixed_frequency)

        while cash_flow_date <= self.maturity:
            # Use a constant period length based on the frequency (e.g., 0.5 years for semiannual)
            time_fraction = np.abs(self.calculate_day_fraction(last_date, cash_flow_date, self.fixed_leg_basis))
            fixed_cash_flow = self.position_type * self.notional * self.fixed_rate * time_fraction
            # if cash_flow_date == self.maturity:
            #     fixed_cash_flow += self.position_type * self.notional

            # Calculate the time since valuation for discounting
            cash_flows.append((cash_flow_date, np.abs(self.calculate_day_fraction(self.valuation_date, cash_flow_date, self.fixed_leg_basis)), fixed_cash_flow, self.fixed_rate))

            # Move to the next payment date
            last_date = cash_flow_date
            cash_flow_date = self.increment_date(cash_flow_date, self.fixed_frequency)
        
        return cash_flows

    def generate_floating_leg_cash_flows(self):
        cash_flows = []

        # self.update_floating_rate(self.valuation_date, self.start_date)
        time_fraction = np.abs(self.calculate_day_fraction(self.valuation_date, self.start_date, self.floating_leg_basis))
        self.accrual_floating = - self.position_type * self.notional * self.floating_rate * time_fraction

        last_date = self.valuation_date
        cash_flow_date = self.increment_date(self.start_date, self.floating_frequency)

        while cash_flow_date <= self.maturity:
            # Update floating rate for the period
            self.update_floating_rate(last_date, cash_flow_date)

            # Use a constant period length based on the frequency (e.g., 1 year for annual)
            time_fraction = np.abs(self.calculate_day_fraction(last_date, cash_flow_date, self.floating_leg_basis))
            floating_cash_flow = - self.position_type * self.notional * self.floating_rate * time_fraction
            # if cash_flow_date == self.maturity:
            #     floating_cash_flow += - self.position_type * self.notional

            # Calculate the time since valuation for discounting
            cash_flows.append((cash_flow_date, np.abs(self.calculate_day_fraction(self.valuation_date, cash_flow_date, self.floating_leg_basis)), floating_cash_flow, self.floating_rate))

            # Move to the next payment date
            last_date = cash_flow_date
            cash_flow_date = self.increment_date(cash_flow_date, self.floating_frequency)
        
        return cash_flows

    def increment_date(self, date, frequency):
        """Helper function to increment date by specified frequency (months)."""
        months = 12 // frequency
        new_month = (date.month + months - 1) % 12 + 1
        new_year = date.year + (date.month + months - 1) // 12

        # Handle end-of-month logic
        if date.day == 31 or (date.month == 2 and (date.day == 28 or date.day == 29)):
            # Check if the new month has fewer days
            last_day_of_new_month = (datetime(new_year, new_month + 1, 1) - timedelta(days=1)).day
            return datetime(new_year, new_month, last_day_of_new_month)

        return datetime(new_year, new_month, min(date.day, 28))
    
    def calculate_npv(self, cash_flows):
        """Calculate NPV using the cash flow schedule."""
        npv = sum(cf / (1 + rate) ** t for _, t, cf, rate in cash_flows)
        return npv

    def calculate_macaulay_duration(self, cash_flows):
        """Calculate the duration using the fixed leg cash flow schedule."""
        weights = [cf / (1 + rate) ** t for _, t, cf, rate in cash_flows]
        duration = sum([w * t for (_, t, _, _), w in zip(cash_flows, weights)]) / sum(weights)
        return duration

    def calculate_convexity(self, cash_flows):
        """Calculate the convexity using the fixed leg cash flow schedule."""
        weights = [cf / (1 + rate) ** t for _, t, cf, rate in cash_flows]
        convexity = sum([w * t ** 2 for (_, t, _, _), w in zip(cash_flows, weights)]) / sum(weights)
        return convexity / 100 # Percentage change in price

    def calculate_dv01(self, cash_flows, basis, bump=0.0001):
        """
        Calculate DV01 for a leg (fixed or floating) by bumping the discount rate.
        
        Parameters:
            cash_flows (list): List of cash flows [(date, time, cash_flow, rate)].
            basis (int): Day count basis used for the cash flows.
            bump (float): Basis point change in yield (default is 1 basis point = 0.0001).
            
        Returns:
            float: DV01 value.
        """
        
        # Calculate NPV with bumped rates
        bumped_cash_flows = [(date, time, cf, rate + bump) for date, time, cf, rate in cash_flows]
        bumped_npv = self.calculate_npv(bumped_cash_flows)

        lessened_cash_flows = [(date, time, cf, rate - bump) for date, time, cf, rate in cash_flows]
        lessened_npv = self.calculate_npv(lessened_cash_flows)
        
        # DV01 is the change in NPV divided by the bump in rate
        dv01 = (bumped_npv - lessened_npv) / 2
        return dv01

    def get_day_count_convention(self, basis):
        """Return the string representation of the day count convention based on the basis."""
        return {
            0: '30/360',
            1: 'Actual/365',
            2: 'Actual/360',
            3: 'Actual/Actual',
            4: '30E/360'
        }.get(basis, 'Unknown')

    def get_info(self):
        info = (
            f"{'='*40}\n"
            f"SWAP INFORMATION\n"
            f"{'='*40}\n"
            f"CUSIP: {self.cusip}\n"
            f"Notional: ${self.notional:,.2f}\n"
            f"Start Date: {self.start_date.strftime('%Y-%m-%d')}\n"
            f"Maturity Date: {self.maturity.strftime('%Y-%m-%d')}\n"
            f"Position: {self.swap_type}\n"
            f"{'-'*40}\n"
        )

        # Fixed Leg Information
        info += (
            f"\nFIXED LEG\n"
            f"{'-'*40}\n"
            f"Fixed Rate: {self.fixed_rate * 100:.2f}%\n"
            f"Fixed Coupon Frequency: {self.fixed_frequency}\n"
            f"Fixed Leg Basis: {self.get_day_count_convention(self.fixed_leg_basis)}\n"
            f"Fixed Leg Principal: ${self.fixed_leg_princpal:,.2f}\n"
            f"Accrual Interest (Fixed): ${self.accrual_fixed:,.2f}\n"
            f"Fixed Leg Value: ${self.fixed_leg_value:,.2f}\n"
            f"Fixed Duration: {self.fixed_duration:.5f}\n"
            f"Fixed DV01: ${self.dv01_fixed:,.5f}\n"
        )
        
        # Display fixed leg cash flows
        info += "Fixed Leg Cash Flows:\n"
        for date, t, cash_flow, rate in self.fixed_leg_cash_flows:
            info += f"  {date.strftime('%Y-%m-%d')}: ${cash_flow:,.2f} at {rate*100:.2f}% with period {t:.2f} years\n"

        # Floating Leg Information
        info += (
            f"\nFLOATING LEG\n"
            f"{'-'*40}\n"
            f"Floating Rate (Latest): {self.floating_rate * 100:.2f}%\n"
            f"Floating Coupon Frequency: {self.floating_frequency}\n"
            f"Floating Leg Basis: {self.get_day_count_convention(self.floating_leg_basis)}\n"
            f"Floating Leg Principal: ${self.floating_leg_principal:,.2f}\n"
            f"Accrual Interest (Floating): ${self.accrual_floating:,.2f}\n"
            f"Floating Leg Value: ${self.floating_leg_value:,.2f}\n"
            f"Floating Duration: {self.floating_duration:.5f}\n"
            f"Floating DV01: ${self.dv01_floating:,.5f}\n"
        )
        
        # Display floating leg cash flows
        info += "Floating Leg Cash Flows:\n"
        for date, t, cash_flow, rate in self.floating_leg_cash_flows:
            info += f"  {date.strftime('%Y-%m-%d')}: ${cash_flow:,.2f} at {rate*100:.2f}% with period {t:.2f} years\n"

        # Overall Swap Information
        info += (
            f"\n{'='*40}\n"
            f"OVERALL SWAP\n"
            f"{'='*40}\n"
            f"Previous Market Value: ${self.previous_market_value:,.2f}\n"
            f"Market Value: ${self.market_value:,.2f}\n"
            f"Mark to Market: ${self.mtm:,.2f}\n"
            f"P&L: ${self.pl:,.2f}\n"
            f"Total Duration: {self.duration:.5f}\n"
            f"Total DV01: ${self.dv01:,.5f}\n"
            f"accrual Interest (Total): ${self.accrual:,.2f}\n"
            f"Convexity: {self.convexity:.5f}\n"
            f"{'='*40}\n"
        )
        
        return info
