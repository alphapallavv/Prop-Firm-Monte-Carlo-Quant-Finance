# monte_carlo.py
import numpy as np
import pandas as pd

def returns_from_csv(df):
    """Try to extract close prices and compute daily returns.
       Accepts TradingView exported CSV where a 'Close' or 'close' column exists.
    """
    possible_cols = ['Close', 'close', 'CLOSE', 'last', 'price']
    col = None
    for c in possible_cols:
        if c in df.columns:
            col = c
            break
    if col is None:
        # fallback: find numeric column besides index if only OHLC present
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) >= 1:
            col = numeric[-1]
        else:
            raise ValueError("Couldn't detect price column. CSV must include a 'Close' or numeric price column.")
    prices = df[col].astype(float).dropna()
    returns = prices.pct_change().dropna()
    return returns

def simulate_bootstrap_paths(returns_series, params):
    """
    Vectorized bootstrap Monte Carlo.
    returns_series : pd.Series of historical returns (daily fractional returns, e.g. 0.01 for 1%)
    params : dict with keys:
      - starting_equity
      - target_profit (absolute $)
      - max_days
      - daily_loss_limit (abs $)
      - static_drawdown_limit (abs $)
      - trailing_drawdown_limit (abs $)
      - n_sims
      - seed (optional)
    Returns:
      dict with keys: equity_matrix (n_sims x max_days), pass_mask (n_sims boolean),
                     pass_day (n_sims int; -1 if not passed), final_equity (n_sims)
    """
    np.random.seed(params.get("seed", 42))
    n_sims = int(params.get("n_sims", 5000))
    max_days = int(params.get("max_days", 180))
    start = float(params.get("starting_equity", 50000.0))
    daily_loss_limit = float(params.get("daily_loss_limit", 1000.0))
    static_dd = float(params.get("static_drawdown_limit", 2000.0))
    trailing_dd = float(params.get("trailing_drawdown_limit", 2000.0))
    target_profit = float(params.get("target_profit", 3000.0))

    # Convert returns to numpy array
    hist_returns = np.array(returns_series)
    if len(hist_returns) == 0:
        raise ValueError("Historic returns series is empty; cannot simulate.")

    # For each sim, sample returns with replacement for max_days
    # To be memory efficient: sample integers and then index
    idx = np.random.randint(0, len(hist_returns), size=(n_sims, max_days))
    sampled = hist_returns[idx]  # shape (n_sims, max_days)

    # Convert to multiplicative returns: 1 + r
    multipliers = 1.0 + sampled

    # Compute equity matrix: cumulative product across days * starting equity
    equity_matrix = start * np.cumprod(multipliers, axis=1)

    # Prepend starting equity as day 0 if needed (for plotting convenience)
    # But we keep equity_matrix shape (n_sims, max_days)
    # Compute daily P&L relative to previous day to test daily loss
    prev = np.hstack([np.full((n_sims,1), start), equity_matrix[:, :-1]])
    daily_pnl = equity_matrix - prev

    # Compute maximum drawdown tracking for static drawdown and trailing drawdown
    running_max = np.maximum.accumulate(equity_matrix, axis=1)
    drawdown_from_peak = running_max - equity_matrix  # positive drawdown numbers

    # Evaluate constraints and pass condition per simulation
    pass_mask = np.zeros(n_sims, dtype=bool)
    pass_day = -1 * np.ones(n_sims, dtype=int)
    fail_reason = np.array(['ok'] * n_sims, dtype=object)

    # Evaluate daily-loss breach: if any day pnl <= -daily_loss_limit -> fail
    daily_loss_breach = (daily_pnl <= -daily_loss_limit)
    daily_loss_any = daily_loss_breach.any(axis=1)

    # Evaluate static drawdown breach: if equity ever falls below start - static_dd
    static_breach = (equity_matrix <= (start - static_dd))
    static_any = static_breach.any(axis=1)

    # Evaluate trailing drawdown: if drawdown_from_peak > trailing_dd ever -> fail
    trailing_breach = (drawdown_from_peak > trailing_dd)
    trailing_any = trailing_breach.any(axis=1)

    # Evaluate target reached day: first day equity >= start + target_profit
    target_level = start + target_profit
    reached_target = (equity_matrix >= target_level)
    # Find first day index where reached (0-based day index)
    first_reach_idx = np.argmax(reached_target, axis=1)
    # If never reached, argmax returns 0; need to detect those
    never_reached = ~reached_target.any(axis=1)
    first_reach_idx[never_reached] = -1

    # Now decide pass: reached target before any failure and before max_days
    for i in range(n_sims):
        if first_reach_idx[i] == -1:
            # no pass
            pass_mask[i] = False
            pass_day[i] = -1
            # find which failure happened first (if any)
            # daily loss
            d_idx = np.where(daily_loss_breach[i])[0]
            s_idx = np.where(static_breach[i])[0]
            t_idx = np.where(trailing_breach[i])[0]
            # earliest failure day if any
            all_fail_days = np.concatenate([d_idx, s_idx, t_idx])
            if all_fail_days.size > 0:
                earliest = int(all_fail_days.min())
                if earliest in d_idx:
                    fail_reason[i] = "daily_loss"
                elif earliest in s_idx:
                    fail_reason[i] = "static_dd"
                else:
                    fail_reason[i] = "trailing_dd"
            else:
                fail_reason[i] = "no_target"
        else:
            # reached a target day; check if any failure occurred before that day
            day = int(first_reach_idx[i])
            # any failure before day?
            if daily_loss_breach[i, :day+1].any() or static_breach[i, :day+1].any() or trailing_breach[i, :day+1].any():
                pass_mask[i] = False
                pass_day[i] = -1
                # determine first failure
                d_idx = np.where(daily_loss_breach[i])[0]
                s_idx = np.where(static_breach[i])[0]
                t_idx = np.where(trailing_breach[i])[0]
                all_fail_days = np.concatenate([d_idx, s_idx, t_idx])
                earliest = int(all_fail_days.min())
                if earliest in d_idx:
                    fail_reason[i] = "daily_loss"
                elif earliest in s_idx:
                    fail_reason[i] = "static_dd"
                else:
                    fail_reason[i] = "trailing_dd"
            else:
                pass_mask[i] = True
                pass_day[i] = day
                fail_reason[i] = "pass"

    results = {
        "equity_matrix": equity_matrix,   # shape (n_sims, max_days)
        "pass_mask": pass_mask,
        "pass_day": pass_day,
        "final_equity": equity_matrix[:, -1],
        "daily_pnl": daily_pnl,
        "fail_reason": fail_reason
    }
    return results
