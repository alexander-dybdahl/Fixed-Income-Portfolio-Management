import copy
import pandas as pd

def process_time_step(portfolio,
                      yield_curve, 
                      current_date, 
                      previous_date, 
                      sheet_data,
                      return_history, 
                      rebalance_strategy=None, 
                      strategy_args=None):
    """
    Process a single time step in the portfolio analysis.

    Parameters:
        portfolio (Portfolio): The portfolio to analyze and update.
        yield_curve (YieldCurve): The yield curve object for yield changes.
        current_date (datetime): The current date for this step.
        previous_date (datetime): The previous date for the analysis.
        sheet_data (DataFrame): The data for the current date (prices and details).
        previous_market_value (float): Market value from the previous step.
        previous_pl (float): P&L from the previous step.
        return_history (dict): Dictionary of historical returns with dates as keys and returns as values.
        risk_free_rate (float): Risk-free rate for calculating Sharpe/Sortino ratios.
        rebalance_strategy (callable): Optional strategy for rebalancing or modifying the portfolio.

    Returns:
        dict: Updated metrics for the portfolio.
    """
    portfolio_copy = copy.deepcopy(portfolio)

    if isinstance(previous_date, str):
            previous_date = pd.to_datetime(previous_date)

    # Optional rebalancing or strategy modification
    strategy_cost = 0
    if rebalance_strategy is not None:
        portfolio, strategy_cost = rebalance_strategy(portfolio, previous_date, current_date, **(strategy_args or {}))

    # Update swaps for the current time step
    for swap in portfolio.swaps:
        swap.update(new_curve_date=current_date, new_valuation_date=current_date)

    # Filter and update portfolio prices
    price_data = sheet_data[sheet_data["CUSIP"].isin([bond.cusip for bond in portfolio.bonds])][["CUSIP", "Market_mid_price"]]
    price_dict = dict(zip(price_data["CUSIP"], price_data["Market_mid_price"]))
    portfolio.update_all_prices({cusip: (current_date.strftime("%Y-%m-%d"), price) for cusip, price in price_dict.items()})
    
    # Calculate yield curve changes
    yield_changes = yield_curve.calculate_yield_change(previous_date, current_date)
    yield_change_at_duration = yield_curve.calculate_yield_change(previous_date, current_date, maturity=portfolio.total_duration)

    # Update portfolio return
    portfolio_return = portfolio.calculate_one_period_return(portfolio_copy.total_market_value)
    return_history[current_date] = portfolio_return

    # Predict price changes using duration and convexity
    yield_impact = portfolio.calculate_yield_impact(yield_change_at_duration)
    convexity_impact = portfolio.calculate_convexity_impact(yield_change_at_duration)
    predicted_price_change = yield_impact + convexity_impact
    actual_price_change = portfolio.market_value + strategy_cost - portfolio_copy.market_value
    residual_impact = actual_price_change - predicted_price_change

    # Calculate Portfolio Metrics
    risk_free_rate = yield_curve.get_nss_yield(current_date, portfolio.total_duration)
    sharpe_ratio = portfolio.calculate_sharpe_ratio(risk_free_rate, return_history)
    sortino_ratio = portfolio.calculate_sortino_ratio(risk_free_rate, return_history)
    cvar = portfolio.calculate_cvar(return_history)

    # Plot the yield curve with bond and portfolio durations
    yield_curve.plot_curve_with_durations_and_price_change([previous_date, current_date], portfolio, use_nss=True)

    # Enhanced Structured Printout with Side-by-Side Comparison
    # print("=" * 80)
    print(f"{'Portfolio Performance Report':^50}")
    print("=" * 80)

    # Header for Comparison Table
    print(f"{'Metric':<24}{'Previous':>15}{'Current':>15}{'Difference':>15}")
    print("=" * 80)

    # Side-by-Side Comparison of Values
    print(f"{'Market Value:':<24}${portfolio_copy.total_market_value:>14,.2f}    ${portfolio.total_market_value:>14,.2f}    ${portfolio.total_market_value - portfolio_copy.total_market_value:>14,.2f}")
    print(f"{'Cash Balance:':<24}${portfolio_copy.cash_balance:>14,.2f}    ${portfolio.cash_balance:>14,.2f}    ${portfolio.cash_balance - portfolio_copy.cash_balance:>14,.2f}")
    print(f"{'Total Value:':<24}${portfolio_copy.total_market_value:>14,.2f}    ${portfolio.total_market_value:>14,.2f}    ${portfolio.total_market_value - portfolio_copy.total_market_value:>14,.2f}")

    # Section: Returns and P&L
    print("=" * 80)
    print(f"{'Return:':<40}{portfolio_return:.2%}")
    # print(f"{'Growth (since inception):':<40}{portfolio.pl / portfolio_copy.market_value * 100:,.2f}%")
    print(f"{'Mark-to-Market:':<40}${portfolio.mtm:,.2f}")
    print(f"{'Accrual Interest:':<40}${portfolio.accrual - portfolio_copy.accrual:,.2f}")
    print(f"{'Unrealized P&L:':<40}${portfolio.mtm + (portfolio.accrual - portfolio_copy.accrual):,.2f}")
    print(f"{'Realized P&L:':<40}${portfolio.realized_pl - portfolio_copy.realized_pl:,.2f}")
    print(f"{'Strategy Gain/Cost:':<40}${strategy_cost:,.2f}")

    # Section: Risk Metrics
    print("=" * 80)
    print(f"{'Duration:':<40}{portfolio.total_duration:,.2f}")
    print(f"{'DV01:':<40}${portfolio.total_dv01:,.2f}")
    print(f"{'Yield Curve Shift:':<40}{yield_changes:.2%}")
    print(f"{'Yield Change at Duration:':<40}{yield_change_at_duration:.2%}")
    print(f"{'Yield Impact:':<40}${yield_impact:,.2f}")
    print(f"{'Convexity Impact:':<40}${convexity_impact:,.2f}")
    print(f"{'Predicted Price Change:':<40}${predicted_price_change:,.2f}")
    print(f"{'Actual Price Change:':<40}${actual_price_change:,.2f}")
    print(f"{'Residual Impact:':<40}${residual_impact:,.2f}")

    # Section: Performance Metrics
    print("=" * 80)
    print(f"{'Sharpe Ratio:':<40}{sharpe_ratio}")
    print(f"{'Sortino Ratio:':<40}{sortino_ratio}")
    print(f"{'CVAR:':<40}{cvar}")
    # print("=" * 80)


    # Compile metrics
    metrics = {
        "portfolio_market_value": portfolio.market_value,
        "portfolio_pl": portfolio.pl,
        "portfolio_duration": portfolio.total_duration,
        "portfolio_dv01": portfolio.total_dv01,
        "yield_changes": yield_changes,
        "yield_change_at_duration": yield_change_at_duration,
        "yield_impact": yield_impact,
        "convexity_impact": convexity_impact,
        "predicted_price_change": predicted_price_change,
        "actual_price_change": actual_price_change,
        "residual_impact": residual_impact,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "cvar": cvar,
    }

    # Optionally update previous values
    commit_changes = input("Commit changes? (yes/no): ").strip().lower() == "yes"
    if commit_changes:
        # print("Changes committed.")
        return metrics, portfolio, current_date
    else:
        # print("Changes discarded.")
        return metrics, portfolio_copy, previous_date