from bond import Bond
from swap import Swap
from utilities import filter_bonds_from_excel
from datetime import timedelta

def rebalance_portfolio_strategy(portfolio, previous_date, current_date, **kwargs):
    """
    Adjusts the portfolio duration by proportionally increasing or decreasing bond weights.

    Parameters:
        portfolio: The portfolio object to rebalance.
        target_duration: The desired target duration.
        scale_factor: The proportion by which to adjust bond notional values (default is 10%).

    Returns:
        None
    """
    target_duration = kwargs.get("target_duration", 0)  # Default is fully hedged
    scale_factor = kwargs.get("scale_factor", 0.1)

    current_duration = portfolio.total_duration
    old_duration = current_duration
    strategy_cost = 0
    print(f"Rebalancing portfolio: Current duration is {current_duration:.2f}, Target duration is {target_duration:.2f}")

    if current_duration > target_duration:
        print("Decreasing portfolio duration...")
        # Decrease weight of longer-duration bonds and increase shorter-duration bonds
        for bond in sorted(portfolio.bonds, key=lambda x: x.modified_duration, reverse=True):
            if current_duration <= target_duration or bond.modified_duration < target_duration:
                break
            strategy_cost += scale_factor * bond.current_price * bond.num_bonds
            old_notional = bond.notional
            old_num_bonds = bond.num_bonds
            portfolio.rebalance_bond(bond, -scale_factor, current_date)
            current_duration = portfolio.total_duration
            portfolio.update()
            print(f"Reduced weight of bond {bond.cusip} from {old_notional:.2f} to {bond.notional:.2f}")

        for bond in sorted(portfolio.bonds, key=lambda x: x.modified_duration):
            if current_duration <= target_duration or bond.modified_duration > target_duration:
                break
            strategy_cost -= scale_factor * bond.current_price * bond.num_bonds
            old_notional += bond.notional
            portfolio.rebalance_bond(bond, scale_factor, current_date)
            current_duration = portfolio.total_duration
            portfolio.update()
            print(f"Increased weight of bond {bond.cusip} from {old_notional:.2f} to {bond.notional:.2f}")

    elif current_duration < target_duration:
        print("Increasing portfolio duration...")
        # Increase weight of longer-duration bonds and decrease shorter-duration bonds
        for bond in sorted(portfolio.bonds, key=lambda x: x.modified_duration, reverse=True):
            if current_duration >= target_duration or bond.modified_duration < target_duration:
                break
            strategy_cost -= scale_factor * bond.current_price * bond.num_bonds
            old_num_bonds = bond.num_bonds
            portfolio.rebalance_bond(bond, scale_factor, current_date)
            current_duration = portfolio.total_duration
            portfolio.update()
            print(f"Increased weight of bond {bond.cusip} from {old_num_bonds:.2f} to {bond.num_bonds:.2f}")

        for bond in sorted(portfolio.bonds, key=lambda x: x.modified_duration):
            if current_duration >= target_duration or bond.modified_duration > target_duration:
                break
            strategy_cost += scale_factor * bond.current_price * bond.num_bonds
            old_num_bonds = bond.num_bonds
            portfolio.rebalance_bond(bond, -scale_factor, current_date)
            current_duration = portfolio.total_duration
            portfolio.update()
            print(f"Reduced weight of bond {bond.cusip} from {old_num_bonds:.2f} to {bond.num_bonds:.2f}")

    print(f"Rebalanced portfolio duration from {old_duration:.2f} to {current_duration:.2f} to fit target duration of {target_duration:.2f}")

    return portfolio, strategy_cost

def hedging_strategy(portfolio, previous_date, current_date, **kwargs):
    """
    Generalized hedging strategy to manage duration risk using treasuries.

    Parameters:
        portfolio (Portfolio): The portfolio to hedge.
        yield_curve (YieldCurve): The yield curve data.
        current_date (datetime): The current date.
        kwargs (dict): Optional arguments:
            - target_duration (float): The target portfolio duration after hedging.
            - treasury_cusip (str): The CUSIP of the treasury bond to use for hedging.
    """
    target_duration = kwargs.get("target_duration", 0)  # Default is fully hedged
    treasury_cusip = kwargs.get("treasury_cusip")
    initial_date = kwargs.get("initial_date")

    strategy_cost = 0

    if treasury_cusip is None:
        raise ValueError("Treasury CUSIP must be provided for hedging.")

    # Calculate duration gap
    duration_gap = portfolio.total_duration - target_duration

    if duration_gap > 0:
        # Hedge by shorting treasury bonds
        Bond.load_bond_data(("Fixed Income Portfolio 2024 - Copy.xlsx"))
        test_bond = Bond.create_bond_from_cusip(treasury_cusip, previous_date, previous_date, previous_date)
        phi = - (duration_gap * portfolio.market_value) / test_bond.modified_duration
        bond = Bond.create_bond_from_cusip(treasury_cusip, previous_date, previous_date, previous_date, notional=phi)
        strategy_cost -= bond.market_value
        portfolio.add_bond(bond, current_date)
        portfolio.update()
        print(f"Hedged portfolio duration risk by shorting {-phi:.2f} number of Treasury bond {treasury_cusip}.")

    return portfolio, strategy_cost

def hedging_strategy_with_swaps(portfolio, previous_date, current_date, **kwargs):
    """Hedge duration using a specific swap."""
    target_duration = kwargs.get("target_duration", 0)  # Default is fully hedged
    swap_cusip = kwargs.get("swap_cusip", f"SWAP-{current_date.strftime('%Y%m%d')}")

    duration_gap = portfolio.total_duration - target_duration

    if duration_gap > 0:
        # Use the swap for hedging
        swap = next((s for s in portfolio.swaps if s.cusip == swap_cusip), None)
        if swap:
            hedge_notional = -(duration_gap * portfolio.total_market_value) / swap.modified_duration
            swap.notional += hedge_notional
            swap.update()
            portfolio.update_all_swaps({swap.cusip: swap.market_value})
            print(f"Hedged duration gap using {swap_cusip}, adjusted notional by {hedge_notional:,.2f}.")

def butterfly_strategy(portfolio, previous_date, current_date, **kwargs):
    """
    Butterfly strategy to balance bond durations (short and long wings, medium body).

    Parameters:
        portfolio (Portfolio): The portfolio to adjust.
        yield_curve (YieldCurve): The yield curve data.
        current_date (datetime): The current date.
        kwargs (dict): Strategy-specific arguments:
            - wing_bonds (list of str): CUSIPs of bonds for the wings (short/long duration).
            - body_bonds (list of str): CUSIPs of bonds for the body (medium duration).
            - wing_weight (float): Weight allocated to the wings.
            - body_weight (float): Weight allocated to the body.
            - settlement_date (str): Settlement date for the bonds.

    Returns:
        Portfolio: Updated portfolio with butterfly strategy applied.
        float: Strategy cost/gain.
    """
    wing_bonds_cusips = kwargs.get("wing_bonds", [])
    body_bonds_cusips = kwargs.get("body_bonds", [])
    investment = kwargs.get("investment")
    wing_weight = kwargs.get("wing_weight", 0.25)
    body_weight = kwargs.get("body_weight", 0.50)
    positive = kwargs.get("positive_butterfly", True)

    if body_weight != 2 * wing_weight:
        raise ValueError("Body weight should be twice the wing weight for a standard butterfly strategy.")

    strategy_cost = 0

    # Load bond data
    Bond.load_bond_data("Fixed Income Portfolio 2024 - Copy.xlsx")

    if not positive: # Negative Butterfly strategy
        investment *= -1

    # Long positions in wing bonds (short and long durations)
    for cusip in wing_bonds_cusips:
        bond = Bond.create_bond_from_cusip(cusip, previous_date, previous_date, previous_date, notional=-investment * wing_weight / len(wing_bonds_cusips))
        strategy_cost -= bond.market_value
        portfolio.add_bond(bond, current_date)

    # Short positions in body bonds (medium durations)
    for cusip in body_bonds_cusips:
        bond = Bond.create_bond_from_cusip(cusip, previous_date, previous_date, previous_date, notional=investment * body_weight / len(body_bonds_cusips))
        strategy_cost -= bond.market_value
        portfolio.add_bond(bond, current_date)

    portfolio.update()
    return portfolio, strategy_cost

def barbell_strategy(portfolio, previous_date, current_date, **kwargs):
    """
    Barbell strategy to focus on short- and long-duration bonds.

    Parameters:
        portfolio (Portfolio): The portfolio to adjust.
        yield_curve (YieldCurve): The yield curve data.
        current_date (datetime): The current date.
        kwargs (dict): Strategy-specific arguments:
            - short_bonds (list of str): CUSIPs of bonds for short durations.
            - long_bonds (list of str): CUSIPs of bonds for long durations.
            - short_weight (float): Weight allocated to short-duration bonds.
            - long_weight (float): Weight allocated to long-duration bonds.
            - initial_date (str): Initial bond prices' date.
            - settlement_date (str): Settlement date for the bonds.

    Returns:
        Portfolio: Updated portfolio with barbell strategy applied.
        float: Strategy cost/gain.
    """
    short_bonds_cusips = kwargs.get("short_bonds", [])
    long_bonds_cusips = kwargs.get("long_bonds", [])
    short_weight = kwargs.get("short_weight", 0.5)
    long_weight = kwargs.get("long_weight", 0.5)
    investment = kwargs.get("investment")

    if short_weight + long_weight > 1:
        raise ValueError("Total weights for short and long bonds cannot exceed 100% of investable amount.")

    strategy_cost = 0

    # Load bond data
    Bond.load_bond_data("Fixed Income Portfolio 2024 - Copy.xlsx")

    # Long positions in short-duration bonds
    for cusip in short_bonds_cusips:
        bond = Bond.create_bond_from_cusip(cusip, previous_date, previous_date, previous_date, notional=investment * short_weight / len(short_bonds_cusips))
        strategy_cost -= bond.market_value
        portfolio.add_bond(bond, current_date)

    # Long positions in long-duration bonds
    for cusip in long_bonds_cusips:
        bond = Bond.create_bond_from_cusip(cusip, previous_date, previous_date, previous_date, notional=investment * long_weight / len(long_bonds_cusips))
        strategy_cost -= bond.market_value
        portfolio.add_bond(bond, current_date)

    portfolio.update()
    return portfolio, strategy_cost

def bullet_strategy(portfolio, previous_date, current_date, **kwargs):
    """
    Concentrates investments around a target duration using pre-computed maturities.

    Parameters:
        portfolio: The portfolio object to adjust.
        current_date: The current date for rebalancing.
        cusip_maturity_dict: Dictionary where keys are maturity dates (YYYY-MM-DD) and values are lists of CUSIPs.
        kwargs: Optional arguments:
            - target_duration: The desired target duration for the bullet strategy.
            - scale_factor: Fraction of the portfolio value to reallocate (default 10%).
            - max_deviation: Maximum allowable deviation from the target duration (default ±0.5 years).
            - investment: Total investment amount for the portfolio.
            - settlement_date: Settlement date for added bonds (defaults to current_date).
            - initial_date: Initial date for bond prices (defaults to current_date).

    Returns:
        portfolio: Updated portfolio.
        strategy_cost: The cost of implementing the strategy.
    """
    target_duration = kwargs.get("target_duration")
    bond_list = kwargs.get("bond_list")
    scale_factor = kwargs.get("scale_factor", 0.1)
    max_deviation = kwargs.get("max_deviation", 0.5)
    investment = kwargs.get("investment", 500_000_000)
    short_bullet = kwargs.get("short_bullet", False)

    if target_duration is None:
        raise ValueError("Target duration must be provided for the bullet strategy.")

    strategy_cost = 0

    # Filter bonds in the portfolio matching target duration
    eligible_bonds = [
        bond for bond in portfolio.bonds
        if abs(abs(bond.modified_duration) - target_duration) <= max_deviation
    ]

    for bond in portfolio.bonds[:]:
        if bond not in eligible_bonds:
            # Remove bond
            strategy_cost += bond.market_value
            portfolio.remove_bond(bond.cusip, current_date)
            print(f"Sold bond {bond.cusip} with duration {bond.modified_duration:.2f} for ${bond.market_value:,.2f}")
        else:
            # Adjust notional by scale factor
            strategy_cost -= scale_factor * bond.market_value
            old_notional = bond.notional
            portfolio.rebalance_bond(bond, scale_factor, current_date)
            print(f"Increase weight of bond {bond.cusip} from ${old_notional:,.2f} to ${bond.notional:,.2f}")

    # Load bond data
    Bond.load_bond_data("Fixed Income Portfolio 2024 - Copy.xlsx")

    if bond_list:
        for cusip in bond_list:
            bond = Bond.create_bond_from_cusip(cusip, previous_date, previous_date, previous_date, notional=investment * scale_factor / len(bond_list))
            strategy_cost -= bond.market_value
            portfolio.add_bond(bond, current_date)

    portfolio.update()
    return portfolio, strategy_cost

def flattener_strategy(portfolio, previous_date, current_date, **kwargs):
    """
    Implements the flattener strategy by selling or shorting long-term bonds.

    Parameters:
        portfolio (Portfolio): The portfolio to adjust.
        yield_curve (YieldCurve): The yield curve object.
        current_date (datetime): The current date for rebalancing.
        kwargs (dict): Optional strategy-specific parameters:
            - short_long_bonds (bool): Whether to short long-term bonds. Defaults to True.
            - target_long_duration (float): Duration threshold for long-term bonds. Defaults to 15 years.
            - scale_factor (float): Proportion to adjust bond holdings. Defaults to 1.0.

    Returns:
        tuple: Updated portfolio and the strategy cost.
    """
    short_long_bonds = kwargs.get("short_long_bonds", True)
    target_long_duration = kwargs.get("target_long_duration", 15.0)

    strategy_cost = 0

    print(f"Executing Flattener Strategy (Target Duration ≥ {target_long_duration:.2f} years)")

    for bond in portfolio.bonds[:]:
        if abs(bond.modified_duration) >= target_long_duration:     
            if short_long_bonds: 
                if bond.notional > 0:
                     # Short the long bond
                    strategy_cost += 2 * abs(bond.market_value)
                    old_notional = bond.notional
                    portfolio.rebalance_bond(bond, -2, current_date)
                    # print(f"Decreased weight of bond {bond.cusip} from ${old_notional:,.2f} to ${bond.notional:,.2f}")
                else:
                    # Short the more of the short bond
                    strategy_cost += abs(bond.market_value)
                    old_notional = bond.notional
                    bond.notional
                    portfolio.rebalance_bond(bond, 1, current_date)
                    # print(f"Increase weight of bond {bond.cusip} from ${old_notional:,.2f} to ${bond.notional:,.2f}")
            else:
                # Sell the bond
                strategy_cost += abs(bond.market_value)
                portfolio.remove_bond(bond.cusip, current_date)
                # print(f"Sold bond {bond.cusip} with duration {bond.modified_duration:.2f} for ${bond.market_value:,.2f}")
        else:
            if bond.notional > 0:
                # Buy more of long the bond
                strategy_cost -= abs(bond.market_value)
                old_notional = bond.notional
                portfolio.rebalance_bond(bond, 1, current_date)
                # print(f"Increase weight of bond {bond.cusip} from ${old_notional:,.2f} to ${bond.notional:,.2f}")
            else:
                # Buy back the short bond
                strategy_cost -= 2 * abs(bond.market_value)
                old_notional = bond.notional
                portfolio.rebalance_bond(bond, -2, current_date)
                # print(f"Increase weight of bond {bond.cusip} from ${old_notional:,.2f} to ${bond.notional:,.2f}")

    portfolio.update()
    print(f"Flattener Strategy executed. Total Strategy Cost: ${strategy_cost:,.2f}")

    return portfolio, strategy_cost

def steepener_rebalance_strategy(portfolio, previous_date, current_date, **kwargs):
    target_weight_long = kwargs.get("target_weight_long", 0.6)
    target_weight_short = kwargs.get("target_weight_short", 0.4)
    scale_factor = kwargs.get("scale_factor", 0.1)

    long_bonds = sorted([bond for bond in portfolio.bonds if bond.modified_duration >= 10], key=lambda x: x.modified_duration)
    short_bonds = sorted([bond for bond in portfolio.bonds if bond.modified_duration < 5], key=lambda x: x.modified_duration)

    total_market_value = portfolio.market_value
    strategy_cost = 0

    # Adjust long-duration bonds
    for bond in long_bonds:
        target_value = total_market_value * target_weight_long / len(long_bonds)
        adjustment = scale_factor * (target_value - bond.market_value)
        portfolio.rebalance_bond(bond, adjustment / bond.market_value, current_date)
        strategy_cost += adjustment

    # Adjust short-duration bonds
    for bond in short_bonds:
        target_value = total_market_value * target_weight_short / len(short_bonds)
        adjustment = scale_factor * (target_value - bond.market_value)
        portfolio.rebalance_bond(bond, -adjustment / bond.market_value, current_date)
        strategy_cost -= adjustment

    return portfolio, strategy_cost

def sell_and_reinvest_strategy(portfolio, previous_date, current_date, selected_cusip=None, **kwargs):
    """
    A strategy that sells all current bonds in the portfolio and reinvests proceeds into a selected bond.

    Parameters:
        portfolio (Portfolio): The portfolio to manage.
        yield_curve (YieldCurve): The yield curve object for pricing and analysis.
        current_date (datetime): The current date for this strategy.
        selected_cusip (str): The CUSIP of the bond to reinvest in.
        kwargs: Additional parameters like initial_date or settlement_date.

    Returns:
        portfolio: Updated portfolio after selling all and reinvesting.
        strategy_cost: The cost or proceeds from implementing the strategy.
    """
    if not selected_cusip:
        raise ValueError("A selected CUSIP must be provided for reinvestment.")

    initial_date = kwargs.get("initial_date")

    # Sell all existing bonds
    total_proceeds = 0
    for bond in portfolio.bonds[:]:  # Use slicing to avoid modifying the list while iterating
        total_proceeds += bond.market_value
        portfolio.remove_bond(bond, current_date)

    print(f"Sold all bonds in the portfolio. Total proceeds: ${total_proceeds:,.2f}")

    # Load the selected bond
    from bond import Bond  # Ensure the Bond class is imported
    Bond.load_bond_data("Fixed Income Portfolio 2024 - Copy.xlsx")
    selected_bond = Bond.create_bond_from_cusip(selected_cusip, previous_date, previous_date, previous_date)

    # Calculate the number of bonds to purchase
    num_bonds_to_buy = total_proceeds / selected_bond.current_price
    selected_bond.notional = num_bonds_to_buy * 100
    selected_bond.num_bonds = num_bonds_to_buy
    selected_bond.update()

    # Add the selected bond to the portfolio
    portfolio.add_bond(selected_bond, current_date)
    portfolio.update()

    print(f"Reinvested proceeds into bond {selected_cusip}. Purchased {num_bonds_to_buy:.2f} units.")
    return portfolio, -total_proceeds
