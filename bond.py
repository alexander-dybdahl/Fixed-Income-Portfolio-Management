import QuantLib as ql
import pandas as pd
from datetime import datetime
from datetime import timedelta

class Bond:
    bond_data = None  # Shared across all Bond instances to hold Excel data

    def __init__(self, cusip, 
                 maturity, 
                 coupon, 
                 bond_name, 
                 country, 
                 industry_sector, 
                 industry_group, 
                 rating, 
                 initial_price,
                 initial_date,
                 notional=100, 
                 frequency=2, 
                 basis=0,
                 prev_coupon_date=None, 
                 next_coupon_date=None, 
                 settlement_date=None):
        
        self.cusip = cusip
        self.maturity = Bond.validate_and_parse_date(maturity, "maturity")
        self.coupon = coupon / 100
        self.bond_name = bond_name
        self.country = country
        self.industry_sector = industry_sector
        self.industry_group = industry_group
        self.rating = rating

        self.initial_price = initial_price
        self.initial_date = Bond.validate_and_parse_date(initial_date, "initial_date")
        
        self.prices = {}

        self.current_price = initial_price
        self.current_price_date = self.initial_date

        self.previous_price = 0
        self.previous_num_bonds = 0
        self.previous_price_date = self.initial_date

        self.notional = notional
        self.frequency = frequency
        self.basis = basis

        self.prev_coupon_date = Bond.validate_and_parse_date(prev_coupon_date, "prev_coupon_date")
        self.next_coupon_date = Bond.validate_and_parse_date(next_coupon_date, "next_coupon_date")
        self.settlement_date = Bond.validate_and_parse_date(settlement_date, "settlement_date")

        self.num_bonds = self.notional / self.initial_price

        # Initialize calculations
        self.update()

    def update(self):
        """Calculate and update bond."""

        self.market_value = self.current_price * self.num_bonds
        self.cash_flows = self.generate_cash_flows()
        self.yield_to_maturity = self.calculate_yield()
        
        self.days_in_period = self.calculate_day_difference(self.prev_coupon_date, self.next_coupon_date)
        self.days_since_last_price = self.calculate_day_difference(self.previous_price_date, self.current_price_date)

        self.daily_accrual = (self.coupon / (self.frequency * self.days_in_period)) * self.notional
        self.accrual = self.daily_accrual * self.days_since_last_price
        
        self.mtm = self.market_value - self.previous_price * self.previous_num_bonds
        self.pl = self.mtm + self.accrual
        self.dirty_price = self.current_price + self.accrual / self.num_bonds

        self.macaulay_duration = self.calculate_macaulay_duration()
        self.modified_duration = self.macaulay_duration / (1 + self.yield_to_maturity)
        self.dollar_duration = self.modified_duration * self.market_value / 100 # Reflects 1% change in yield
        self.dv01 = self.dollar_duration / 100 # Reflects 1 basis point (0.01%) change
        self.convexity = self.calculate_convexity() # Percentage change in price

        self.approximate_dv01 = self.calculate_approximate_dv01() # Reflects 1 basis point (0.01%) change
        self.approximate_convexity = self.calculate_approximate_convexity() # Percentage change in price

    @staticmethod
    def validate_and_parse_date(date_value, date_name):
        """
        Validate and parse the input date.
        Parameters:
            date_value: The input date value to be validated and parsed.
            date_name: The name of the date field for error messages.
        Returns:
            datetime: A datetime object if the input is valid; None if the input is None.
        """
        if date_value is None:
            return None  # Allow None values for optional dates
        if isinstance(date_value, datetime):
            return date_value  # Already a datetime object
        try:
            # Attempt to parse from string, ignoring any time components
            return datetime.strptime(str(date_value).split()[0], "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid {date_name}: {date_value}. Ensure the date is in 'YYYY-MM-DD' format.")

    @classmethod
    def load_bond_data(cls, file_path):
        """
        Load bond data from an Excel file with multiple sheets, skipping the first two sheets.
        
        Parameters:
            file_path (str): Path to the Excel file.
        """
        excel_data = pd.ExcelFile(file_path)
        sheets_to_load = excel_data.sheet_names[2:]  # Skip the first two sheets
        all_data = []

        for sheet_name in sheets_to_load:
            sheet_data = excel_data.parse(sheet_name)
            sheet_data["Date"] = pd.to_datetime(sheet_name, format="%B %d %Y")  # Parse sheet name as date
            all_data.append(sheet_data)

        # Combine all sheets into a single DataFrame
        cls.bond_data = pd.concat(all_data, ignore_index=True)

    @classmethod
    def create_bond_from_cusip(cls, cusip, current_date, initial_date, settlement_date, notional=100):
        """
        Create a Bond instance based on its CUSIP and historical data.

        Parameters:
            cusip (str): The CUSIP of the bond to create.
            current_date (str or datetime): The current date up to which prices should be added.
            initial_date (str or datetime, optional): Initial date for the bond. Defaults to None.
            settlement_date (str or datetime, optional): Settlement date for the bond. Defaults to None.

        Returns:
            Bond: A bond initialized with historical prices up to the current date.
        """
        if cls.bond_data is None:
            raise ValueError("Bond data not loaded. Call `load_bond_data` first.")

        # Ensure `current_date` is a datetime object
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, "%Y-%m-%d")

        # Filter data for the specified CUSIP and up to the current date
        bond_rows = cls.bond_data[
            (cls.bond_data["CUSIP"] == cusip) & (cls.bond_data["Date"] <= current_date) & (cls.bond_data["Date"] >= initial_date)
        ]
        if bond_rows.empty:
            raise ValueError(f"No bond found with CUSIP: {cusip} or no data up to the current date: {current_date}.")


        # Use the first row to initialize the bond
        initial_row = bond_rows.iloc[0]
        frequency_value = initial_row.get("CPN_FREQ", 2)  # Default to 2 if missing
        frequency_value = 2 if pd.isna(frequency_value) else int(frequency_value)

        bond = cls(
            cusip=cusip,
            maturity=initial_row["MATURITY"].strftime("%Y-%m-%d"),
            coupon=initial_row["CPN"],
            bond_name=initial_row["NAME"],
            country=initial_row.get("COUNTRY", "Unknown"),
            industry_sector=initial_row.get("INDUSTRY_SECTOR", "Unknown"),
            industry_group=initial_row.get("INDUSTRY_GROUP", "Unknown"),
            rating=initial_row.get("BB_COMPOSITE", "Unknown"),
            initial_price=initial_row["Market_mid_price"],
            initial_date=initial_date,
            notional=notional,
            frequency=frequency_value,
            basis=0,
            prev_coupon_date=initial_row.get("PREV_CPN_DT").strftime("%Y-%m-%d") if pd.notna(initial_row.get("PREV_CPN_DT")) else None,
            next_coupon_date=initial_row.get("NXT_CPN_DT").strftime("%Y-%m-%d") if pd.notna(initial_row.get("NXT_CPN_DT")) else None,
            settlement_date=settlement_date,
        )

        # Add all historical prices for the bond up to the current date
        for _, row in bond_rows.iterrows():
            bond.update_price(row["Date"].strftime("%Y-%m-%d"), row["Market_mid_price"])

        return bond

    def calculate_day_difference(self, start_date, end_date):
        """Calculate day difference based on day count convention (basis)."""
        if self.basis == 0:  # 30/360 day count convention
            start_day = min(start_date.day, 30)
            end_day = min(end_date.day, 30)
            day_diff = (end_date.year - start_date.year) * 360 + (end_date.month - start_date.month) * 30 + (end_day - start_day)
        elif self.basis == 1:  # Actual/365 day count convention
            day_diff = (end_date - start_date).days
        elif self.basis == 2:  # Actual/360 day count convention
            day_diff = (end_date - start_date).days
        elif self.basis == 3:  # Actual/Actual convention
            day_diff = (end_date - start_date).days
        elif self.basis == 4:  # 30E/360 convention (Eurobond basis)
            start_day = min(start_date.day, 30) if start_date.month != 2 else start_date.day
            end_day = min(end_date.day, 30) if end_date.month != 2 else end_date.day
            day_diff = (end_date.year - start_date.year) * 360 + (end_date.month - start_date.month) * 30 + (end_day - start_day)
        else:
            raise ValueError("Unsupported basis. Use 0 for 30/360, 1 for Actual/365, 2 for Actual/360, 3 for Actual/Actual, or 4 for 30E/360.")
        return day_diff

    def calculate_year_fraction(self, start_date, end_date):
        """Calculate the year fraction based on the day count convention (basis) using day difference."""
        day_diff = self.calculate_day_difference(start_date, end_date)
        
        if self.basis == 0 or self.basis == 4:  # 30/360 or 30E/360
            return day_diff / 360
        elif self.basis == 1:  # Actual/365
            return day_diff / 365
        elif self.basis == 2:  # Actual/360
            return day_diff / 360
        elif self.basis == 3:  # Actual/Actual
            # Adjust for leap years if necessary
            days_in_year = 366 if start_date.year % 4 == 0 else 365
            return day_diff / days_in_year
        else:
            raise ValueError("Unsupported basis. Use 0 for 30/360, 1 for Actual/365, 2 for Actual/360, 3 for Actual/Actual, or 4 for 30E/360.")

    def update_prices_from_data(self):
        """
        Update bond prices using the shared bond_data DataFrame.
        """
        if Bond.bond_data is None:
            raise ValueError("Bond data not loaded. Call `load_bond_data` first.")
        
        # Fetch price history for this bond
        price_history = Bond.bond_data[Bond.bond_data["CUSIP"] == self.cusip]
        for _, row in price_history.iterrows():
            self.update_price(row["Date"], row["Market_mid_price"])

    def update_price(self, date, price):
        """Update bond price and set current price if the new date is the latest."""
        date_obj = datetime.strptime(str(date), "%Y-%m-%d")
        self.prices[date] = price

        # Check if this date is more recent than the current price date
        if date_obj > self.current_price_date:
            self.previous_price = self.current_price
            self.previous_num_bonds = self.num_bonds
            self.previous_price_date = self.current_price_date
            self.current_price = price
            self.current_price_date = date_obj

        self.update()

    def generate_cash_flows(self):
        """Generates the cash flow schedule from settlement date to maturity, including the bond's payment frequency."""
        schedule = []
        cash_flow_date = self.next_coupon_date
        last_date = self.settlement_date

        # Generate coupon payments until maturity
        while cash_flow_date <= self.maturity:
            time_fraction = self.calculate_year_fraction(last_date, cash_flow_date)
            
            # Calculate cash flow for this period
            cash_flow_amount = self.notional * self.coupon * time_fraction
            schedule.append((cash_flow_date, self.calculate_year_fraction(self.settlement_date, cash_flow_date), cash_flow_amount, self.frequency))

            last_date = cash_flow_date
            
            # Adjust cash flow date for the next payment period
            cash_flow_date = self.increment_date(cash_flow_date)

        # Principal repayment at maturity
        if schedule and schedule[-1][0] == self.maturity:
            schedule[-1] = (self.maturity, schedule[-1][1], schedule[-1][2] + self.notional, self.frequency)
        else:
            schedule.append((self.maturity, self.calculate_year_fraction(self.settlement_date, self.maturity), self.notional, self.frequency))
        
        return schedule

    def increment_date(self, date):
        """Helper function to increment date by specified months."""
        months = 12 // self.frequency
        new_month = (date.month + months - 1) % 12 + 1
        new_year = date.year + (date.month + months - 1) // 12
        return datetime(new_year, new_month, min(date.day, 28))
    
    def calculate_yield(self):
        """Calculate the yield to maturity for the bond based on the cash flow schedule."""

        if not self.cash_flows:
            print(f"Invalid cash flows or market value for bond {self.cusip}.")
            return 0  # or fallback value

        # Define the function for NPV
        def npv(rate):
            total_npv = sum(abs(cf) / (1 + rate) ** t for _, t, cf, frequency in self.cash_flows)
            return total_npv - abs(self.market_value)

        # Define the Brent solver
        solver = ql.Brent()

        # Attempt to solve with an initial guess and bounds for the Brent solver
        try:
            yield_to_maturity = solver.solve(npv, 1e-9, 0.05, 0, 1)
        except RuntimeError as e:
            print(f"Error calculating yield for bond {self.cusip} with market value {self.market_value}")
            yield_to_maturity = 0  # Ensure fallback is non-negative

        return yield_to_maturity
    
    def calculate_macaulay_duration(self):
        """Calculate Macaulay duration using the cash flow schedule."""
        cash_flows = self.generate_cash_flows()
        weights = [cf / (1 + self.yield_to_maturity) ** t for _, t, cf, frequency in cash_flows]
        total_weight = sum(weights)
        durations = [w * t for (_, t, _, _), w in zip(cash_flows, weights)]
        return sum(durations) / abs(total_weight)

    def npv(self, rate):
        """Calculate NPV using the cash flow schedule."""
        npv = sum(cf / (1 + rate) ** t for _, t, cf, freq in self.cash_flows)
        return npv

    def calculate_convexity(self):
        """Calculate convexity using the cash flow schedule."""
        weights = [cf / (1 + self.yield_to_maturity) ** t for _, t, cf, frequency in self.cash_flows]
        total_weight = sum(weights)
        convexities = [w * t ** 2 for (_, t, _, _), w in zip(self.cash_flows, weights)]
        return sum(convexities) / total_weight / 100

    def calculate_approximate_dv01(self):
        """Calculate DV01 using finite difference."""
        delta_yield = 0.0001  # 1 basis point shift

        original_yield = self.yield_to_maturity
        price_up = self.npv(original_yield + delta_yield)
        price_down = self.npv(original_yield - delta_yield)
        dv01 = (price_down - price_up) / 2
        return dv01

    def calculate_approximate_convexity(self):
        """Calculate convexity using the numerical method with yield shifts."""
        delta_yield = 0.0001  # 1 basis point shift
        
        original_yield = self.yield_to_maturity
        price_up = self.npv(original_yield + delta_yield)
        price_down = self.npv(original_yield - delta_yield)
        convexity = ((price_up + price_down - 2 * self.market_value) / (self.dirty_price * self.num_bonds * delta_yield ** 2))
        
        return convexity / 100
    



# Printouts


    def print_bond_info(self):
        # Returns bond information as a formatted string
        info = (
            f"CUSIP: {self.cusip}\n"
            f"Bond Name: {self.bond_name}\n"
            f"Country: {self.country}\n"
            f"Industry Sector: {self.industry_sector}\n"
            f"Industry Group: {self.industry_group}\n"
            f"Rating: {self.rating}\n"
            f"Frequency: {self.frequency}\n"
            f"Coupon: {self.coupon*100:.4f}%\n"
            f"Yield to Maturity: {self.yield_to_maturity*100:.3f}%\n"
            f"Maturity: {self.maturity}\n"
            f"Notional: ${self.notional:,.2f}\n"
            f"Market Value: ${self.market_value:,.2f}\n"
            f"Number of Bonds: {int(self.num_bonds)}\n"
            f"Initial Price: ${self.initial_price:.3f}\n"
            f"Current Price: ${self.current_price:.3f}\n"
            f"Macaulay Duration: {self.macaulay_duration:.4f}\n"
            f"Modified Duration: {self.modified_duration:.4f}\n"
            f"Dollar Duration: ${self.dollar_duration:,.5f}\n"
            f"DV01: ${self.dv01:,.5f}\n"
            f"Convexity: %{self.convexity:.5f}\n"
            f"Approximate DV01: ${self.approximate_dv01:,.5f}\n"
            f"Approximate Convexity: %{self.approximate_convexity:.5f}\n"
            f"Days in period: {self.days_in_period}\n"
            f"Daily Accrual: ${self.daily_accrual:.3f}\n"
            f"Accrual (since settlement): ${self.accrual:,.3f}\n"
            f"Dirty Price: ${self.current_price:.3f} + ${self.accrual/self.num_bonds:.3f} = ${self.dirty_price:.3f}\n"
            f"P&L: ${self.pl:,.3f}\n"
        )
        for date, price in self.prices.items():
            info += f"{date}: ${price:.3f} | "
        print(info + "\n" + "-" * 50 + "\n")
