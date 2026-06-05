"""
CVaR-Based Portfolio Optimization  (Linear Programming, Rockafellar-Uryasev)
============================================================================
MSBA Optimization Project 1 - Group 11
Emma Trunnell, Nathan Arimilli, Nikhil Kumar, Satvik Shankar

Builds a linear program that minimizes the Conditional Value-at-Risk (CVaR,
a.k.a. Expected Shortfall) of a long-only stock portfolio, trains on 2019
NASDAQ-100 returns, and stress-tests on the 2020 COVID shock.

This single script reproduces every task end-to-end and writes all figures
and result tables to ../output/.  Run it from anywhere:

    python cvar_portfolio.py

Tasks
-----
2. Baseline beta-CVaR (beta=0.95, R=0.02%/day): train 2019, test 2020, vs NDX.
3. Sensitivity of allocation / risk to the confidence level beta.
4. Minimax: minimize the *worst month's* CVaR in 2019.
5. Monthly re-optimization in 2020 on a rolling 12-month window.
6. Stability of the monthly allocations (<=5 percentage-point moves).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / reproducible figure rendering
import matplotlib.pyplot as plt
from gurobipy import GRB, Model, quicksum

# A blocked-matmul FPE flag is raised spuriously by some macOS BLAS builds even
# on fully finite data; it does not affect any result, so we silence it.
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")
# Two CSVs use different date spellings; we parse defensively and silence the
# pandas "could not infer format" notice (the dates parse correctly either way).
warnings.filterwarnings("ignore", message=".*Could not infer format.*")

# ---------------------------------------------------------------------------
# Paths (relative to this file, so the script runs from any working directory)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model parameters (generalized - NOT hard-coded to a particular data shape)
# ---------------------------------------------------------------------------
BETA = 0.95        # confidence level for the baseline
RMIN = 0.0002      # minimum expected daily return floor (0.02%/day)
BENCHMARK = "NDX"  # index column, excluded from the investable universe

# =====================  >>> CSV INPUTS (graders: edit here) <<<  =============
# The first code section reads the two price CSVs.  To re-grade on new data,
# change ONLY these two paths; everything downstream is computed from them.
TRAIN_CSV = os.path.join(DATA_DIR, "stocks2019.csv")
TEST_CSV = os.path.join(DATA_DIR, "stocks2020.csv")
# ============================================================================


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_returns(prices_train_df, prices_test_df,
                    benchmark_col=BENCHMARK, min_keep=0.95):
    """Convert two price tables into aligned daily-return matrices.

    - First column is treated as the date and used as a sorted index.
    - Prices -> simple daily returns via pct_change.
    - The benchmark (NDX) is split out so it is never an investable asset.
    - Assets missing more than (1 - min_keep) of their observations are dropped;
      small remaining gaps are forward/back-filled.
    - The investable universe is the intersection of both years' columns.

    Returns: r_train, r_test, ndx_train, ndx_test, common_symbols
    """
    def _rets(prices_df):
        df = prices_df.copy()
        date_col = df.columns[0]                       # first column = date
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")  # force numeric prices
        rets = df.pct_change().dropna(how="all")
        # Guard against zero/te-zero prices that would create +/-inf returns.
        rets = rets.replace([np.inf, -np.inf], np.nan)
        bench = rets[benchmark_col].copy() if benchmark_col in rets.columns else None
        if benchmark_col in rets.columns:
            rets = rets.drop(columns=[benchmark_col])
        rets = rets.dropna(axis=1, thresh=int(min_keep * len(rets)))
        rets = rets.ffill().bfill()
        return rets, bench

    r_train, b_train = _rets(prices_train_df)
    r_test, b_test = _rets(prices_test_df)

    cols = sorted(set(r_train.columns) & set(r_test.columns))
    return r_train[cols], r_test[cols], b_train, b_test, cols


# ---------------------------------------------------------------------------
# Core CVaR linear program (Rockafellar-Uryasev linearization)
# ---------------------------------------------------------------------------
def solve_cvar_lp(returns_df, beta=BETA, R=RMIN, verbose=False):
    """Minimize the empirical beta-CVaR of portfolio losses.

    min_{x, alpha, u}   alpha + 1/((1-beta) T) * sum_k u_k
    s.t.  u_k >= -x . y_k - alpha,   u_k >= 0       (tail-loss linearization)
          sum_j x_j = 1                              (budget)
          mu . x   >= R                              (expected-return floor)
          0 <= x_j <= 1                              (long-only, no leverage)
    """
    T, N = returns_df.shape
    assets = list(returns_df.columns)
    Y = returns_df.values                        # scenario matrix (T x N)
    mu = returns_df.mean(axis=0).values          # expected return per asset

    m = Model("cvar_min")
    if not verbose:
        m.Params.OutputFlag = 0

    x = m.addVars(N, lb=0.0, ub=1.0, name="x")          # portfolio weights
    alpha = m.addVar(lb=-GRB.INFINITY, name="alpha")    # VaR variable
    u = m.addVars(T, lb=0.0, name="u")                  # tail-loss slacks

    m.addConstr(quicksum(x[j] for j in range(N)) == 1.0, name="budget")
    m.addConstr(quicksum(mu[j] * x[j] for j in range(N)) >= R, name="min_return")
    for k in range(T):
        m.addConstr(
            u[k] >= -quicksum(Y[k, j] * x[j] for j in range(N)) - alpha,
            name=f"tail_{k}",
        )

    coef = 1.0 / ((1.0 - beta) * T)
    m.setObjective(alpha + coef * quicksum(u[k] for k in range(T)), GRB.MINIMIZE)
    m.optimize()

    x_sol = np.array([x[j].X for j in range(N)])
    return {"x": x_sol, "alpha": alpha.X, "obj": m.ObjVal,
            "assets": assets, "mu": mu, "T": T, "N": N, "beta": beta, "R": R}


def solve_minimax_monthly_cvar(returns_df, beta=BETA, R=RMIN, verbose=False):
    """Minimize the maximum monthly beta-CVaR over the training year.

    One static weight vector x is chosen; a free variable t upper-bounds every
    month's CVaR, and we minimize t.  Each month gets its own alpha_m and slacks.
    """
    df = returns_df.copy()
    months = df.index.to_period("M")
    unique_months = sorted(months.unique())
    month_indices = {mo: np.where(months == mo)[0].tolist() for mo in unique_months}

    T, N = df.shape
    assets = list(df.columns)
    Y = df.values
    mu = df.mean(axis=0).values

    m = Model("minimax_monthly_cvar")
    if not verbose:
        m.Params.OutputFlag = 0

    x = m.addVars(N, lb=0.0, ub=1.0, name="x")
    t = m.addVar(lb=-GRB.INFINITY, name="t")

    m.addConstr(quicksum(x[j] for j in range(N)) == 1.0, name="budget")
    m.addConstr(quicksum(mu[j] * x[j] for j in range(N)) >= R, name="min_return")

    for mi, mo in enumerate(unique_months):
        idxs = month_indices[mo]
        alpha_m = m.addVar(lb=-GRB.INFINITY, name=f"alpha_{mi}")
        u = m.addVars(len(idxs), lb=0.0, name=f"u_{mi}")
        for r, k in enumerate(idxs):
            m.addConstr(u[r] >= -quicksum(Y[k, j] * x[j] for j in range(N)) - alpha_m)
        coef = 1.0 / ((1.0 - beta) * len(idxs))
        m.addConstr(t >= alpha_m + coef * quicksum(u[r] for r in range(len(idxs))))

    m.setObjective(t, GRB.MINIMIZE)
    m.optimize()

    x_sol = np.array([x[j].X for j in range(N)])
    return {"x": x_sol, "assets": assets, "t": t.X}


# ---------------------------------------------------------------------------
# Evaluation  (single, consistent Rockafellar-Uryasev CVaR definition)
# ---------------------------------------------------------------------------
def cvar_var(returns_1d, beta=BETA):
    """Empirical VaR and CVaR (loss domain) using the R-U expected-shortfall.

    VaR  = beta-quantile of the loss distribution (losses = -returns).
    CVaR = VaR + E[ (loss - VaR)_+ ] / (1 - beta).

    This is exactly the quantity the LP minimizes, so the in-sample CVaR of the
    optimal portfolio equals the LP objective; the same formula is used for all
    out-of-sample and benchmark numbers.
    """
    losses = -np.asarray(returns_1d, dtype=float)
    var = float(np.quantile(losses, beta))
    cvar = var + float(np.mean(np.maximum(losses - var, 0.0))) / (1.0 - beta)
    return var, cvar


def evaluate_portfolio(x, returns_df, beta=BETA):
    """Average return, volatility, VaR and CVaR of weights x on a return panel."""
    pr = returns_df.values @ np.asarray(x)
    var, cvar = cvar_var(pr, beta=beta)
    return {"avg_daily_return": float(np.mean(pr)),
            "std_daily_return": float(np.std(pr, ddof=1)),
            "VaR_beta_loss": var, "CVaR_beta_loss": cvar,
            "n_days": int(len(pr)), "port_returns": pr}


def diversification(weights, tol=1e-6):
    """Concentration diagnostics for a weight vector.

    HHI            = sum w_i^2          (1 = all-in-one, 1/N = equal-weight)
    effective_N    = 1 / HHI           (number-equivalent of holdings)
    n_nonzero      = count of holdings above a small tolerance
    """
    w = np.asarray(weights, dtype=float)
    hhi = float(np.sum(w ** 2))
    return {"HHI": hhi, "effective_N": 1.0 / hhi if hhi > 0 else np.nan,
            "n_holdings": int(np.sum(w > tol))}


def _pct(x, p=2):
    return f"{100.0 * float(x):.{p}f}%" if x is not None else "n/a"


# ---------------------------------------------------------------------------
# Figure helper
# ---------------------------------------------------------------------------
def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> output/figures/{name}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    # ---- Load data (the graded read_csv call) ----
    prices_train = pd.read_csv(TRAIN_CSV)
    prices_test = pd.read_csv(TEST_CSV)

    rets_train, rets_test, ndx_train, ndx_test, symbols = prepare_returns(
        prices_train, prices_test
    )
    train_year = rets_train.index.year.min()
    test_year = rets_test.index.year.min()
    print(f"Train {train_year}: {rets_train.shape}  Test {test_year}: {rets_test.shape}")
    print(f"Investable universe: {len(symbols)} assets   "
          f"NaNs remaining: {int(rets_train.isna().sum().sum() + rets_test.isna().sum().sum())}")

    results = {}  # collected headline numbers for the key-results table

    # ===================================================================
    # TASK 2 - Baseline CVaR (train train_year, test test_year, vs NDX)
    # ===================================================================
    print("\n" + "=" * 60 + "\nPART 2: Baseline CVaR\n" + "=" * 60)
    sol2 = solve_cvar_lp(rets_train, beta=BETA, R=RMIN)
    e_in = evaluate_portfolio(sol2["x"], rets_train, beta=BETA)
    e_out = evaluate_portfolio(sol2["x"], rets_test, beta=BETA)

    ndx_in_var, ndx_in_cvar = cvar_var(ndx_train.values, BETA)
    ndx_out_var, ndx_out_cvar = cvar_var(ndx_test.values, BETA)

    part2 = pd.DataFrame(
        {
            f"Optimized {train_year} (in-sample)": [
                e_in["avg_daily_return"], e_in["std_daily_return"],
                e_in["VaR_beta_loss"], e_in["CVaR_beta_loss"]],
            f"Optimized {test_year} (out-of-sample)": [
                e_out["avg_daily_return"], e_out["std_daily_return"],
                e_out["VaR_beta_loss"], e_out["CVaR_beta_loss"]],
            f"NDX {train_year}": [
                float(ndx_train.mean()), float(ndx_train.std(ddof=1)),
                ndx_in_var, ndx_in_cvar],
            f"NDX {test_year}": [
                float(ndx_test.mean()), float(ndx_test.std(ddof=1)),
                ndx_out_var, ndx_out_cvar],
        },
        index=["Avg daily return", "Volatility (std)",
               f"VaR (loss) beta={BETA}", f"CVaR (loss) beta={BETA}"],
    )
    print(part2.map(_pct).to_string())
    part2.map(lambda v: round(v, 6)).to_csv(os.path.join(OUT_DIR, "part2_metrics.csv"))

    top2 = pd.Series(sol2["x"], index=sol2["assets"]).sort_values(ascending=False)
    top2[top2 > 1e-6].to_frame("weight").to_csv(os.path.join(OUT_DIR, "part2_top_holdings.csv"))
    div2 = diversification(sol2["x"])
    print(f"Diversification: {div2['n_holdings']} holdings, "
          f"HHI={div2['HHI']:.3f}, effective N={div2['effective_N']:.1f}")

    results["Task 2 - In-sample CVaR (%s)" % train_year] = e_in["CVaR_beta_loss"]
    results["Task 2 - Out-of-sample CVaR (%s)" % test_year] = e_out["CVaR_beta_loss"]
    results["Task 2 - NDX CVaR (%s)" % test_year] = ndx_out_cvar

    # Figure: CVaR bars (portfolio vs NDX, both years)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = [f"Opt {train_year}", f"Opt {test_year}", f"NDX {train_year}", f"NDX {test_year}"]
    vals = [e_in["CVaR_beta_loss"], e_out["CVaR_beta_loss"], ndx_in_cvar, ndx_out_cvar]
    colors = ["#2c7fb8", "#2c7fb8", "#bdbdbd", "#bdbdbd"]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100, f"{v*100:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"Daily CVaR (loss), beta={BETA}  [%]")
    ax.set_title("Figure 1. Tail risk: optimized portfolio vs NDX benchmark")
    _save(fig, "fig1_part2_cvar_bars.png")

    # Figure: cumulative return trends (portfolio vs NDX), both years  [feedback #1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, (rp, rb, yr) in zip(
        axes,
        [(e_in["port_returns"], ndx_train.values, train_year),
         (e_out["port_returns"], ndx_test.values, test_year)],
    ):
        cum_p = np.cumprod(1 + rp) - 1
        cum_b = np.cumprod(1 + rb) - 1
        idx = rets_train.index if yr == train_year else rets_test.index
        ax.plot(idx, cum_p * 100, label="Optimized portfolio", color="#2c7fb8")
        ax.plot(idx, cum_b * 100, label="NDX", color="#999999", ls="--")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{yr}")
        ax.set_ylabel("Cumulative return [%]")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Figure 2. Cumulative return: optimized portfolio vs NDX")
    fig.tight_layout()
    _save(fig, "fig2_part2_cumulative_returns.png")

    # ===================================================================
    # TASK 3 - Sensitivity to beta
    # ===================================================================
    print("\n" + "=" * 60 + "\nPART 3: Beta Sensitivity\n" + "=" * 60)
    betas = (0.90, 0.95, 0.99)
    rows, allocs, div_rows = [], {}, []
    for b in betas:
        sb = solve_cvar_lp(rets_train, beta=b, R=RMIN)
        ein = evaluate_portfolio(sb["x"], rets_train, beta=b)
        eout = evaluate_portfolio(sb["x"], rets_test, beta=b)
        d = diversification(sb["x"])
        rows.append({"beta": b,
                     f"{train_year} CVaR": ein["CVaR_beta_loss"],
                     f"{test_year} CVaR": eout["CVaR_beta_loss"],
                     f"{train_year} avg ret": ein["avg_daily_return"],
                     f"{test_year} avg ret": eout["avg_daily_return"],
                     "HHI": d["HHI"], "effective_N": d["effective_N"],
                     "n_holdings": d["n_holdings"]})
        div_rows.append(d)
        allocs[b] = pd.Series(sb["x"], index=sb["assets"])
    beta_df = pd.DataFrame(rows).set_index("beta")
    print(beta_df.to_string(float_format=lambda v: f"{v:.4f}"))
    beta_df.to_csv(os.path.join(OUT_DIR, "part3_beta_summary.csv"))
    results["Task 3 - Out-of-sample CVaR @ beta=0.99 (%s)" % test_year] = \
        beta_df.loc[0.99, f"{test_year} CVaR"]

    # Figure: CVaR vs beta (in vs out)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(beta_df.index, beta_df[f"{train_year} CVaR"] * 100, "o-",
            label=f"{train_year} (in-sample)", color="#2c7fb8")
    ax.plot(beta_df.index, beta_df[f"{test_year} CVaR"] * 100, "s-",
            label=f"{test_year} (out-of-sample)", color="#d95f0e")
    for b in betas:
        ax.annotate(f"{beta_df.loc[b, f'{test_year} CVaR']*100:.2f}%",
                    (b, beta_df.loc[b, f"{test_year} CVaR"] * 100),
                    textcoords="offset points", xytext=(0, 6), fontsize=8)
    ax.set_xlabel("Confidence level beta")
    ax.set_ylabel("Daily CVaR (loss) [%]")
    ax.set_title("Figure 3. CVaR vs beta (in-sample stays low, out-of-sample blows up)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fig3_part3_cvar_vs_beta.png")

    # Figure: diversification vs beta  [feedback #3 - quantify diversification]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([f"beta={b}" for b in betas], beta_df["effective_N"].values,
           color="#3182bd")
    for i, b in enumerate(betas):
        ax.text(i, beta_df.loc[b, "effective_N"],
                f"{beta_df.loc[b,'effective_N']:.1f}\n({beta_df.loc[b,'n_holdings']} held)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Effective number of holdings  (1 / HHI)")
    ax.set_title("Figure 4. Diversification falls as beta rises (more tail-focused)")
    _save(fig, "fig4_part3_diversification.png")

    # Figure: top-weights small multiples
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, b in zip(axes, betas):
        s = allocs[b].sort_values(ascending=False).head(8)
        ax.bar(s.index, s.values * 100, color="#2c7fb8")
        ax.set_title(f"beta={b}  (eff. N={beta_df.loc[b,'effective_N']:.1f})")
        ax.set_ylabel("Weight [%]")
        ax.tick_params(axis="x", rotation=60, labelsize=8)
    fig.suptitle("Figure 5. Allocation concentrates into XEL/CHTR as beta increases")
    fig.tight_layout()
    _save(fig, "fig5_part3_weights_by_beta.png")

    # ===================================================================
    # TASK 4 - Minimax monthly CVaR (train year)
    # ===================================================================
    print("\n" + "=" * 60 + "\nPART 4: Minimax Monthly CVaR\n" + "=" * 60)
    sol4 = solve_minimax_monthly_cvar(rets_train, beta=BETA, R=RMIN)
    e4_out = evaluate_portfolio(sol4["x"], rets_test, beta=BETA)
    div4 = diversification(sol4["x"])
    print(f"{train_year} worst-month CVaR (minimax obj): {_pct(sol4['t'])}")
    print(f"{test_year} daily CVaR of minimax weights:   {_pct(e4_out['CVaR_beta_loss'])}")
    print(f"Diversification: {div4['n_holdings']} holdings, "
          f"effective N={div4['effective_N']:.1f}")
    results["Task 4 - Worst-month CVaR (%s, train)" % train_year] = sol4["t"]
    results["Task 4 - Out-of-sample daily CVaR (%s)" % test_year] = e4_out["CVaR_beta_loss"]

    # Quantitative similarity: minimax vs the high-beta (0.99) portfolio  [feedback #2]
    w_mm = pd.Series(sol4["x"], index=sol4["assets"])
    w_b99 = allocs[0.99].reindex(w_mm.index).fillna(0.0)
    w_b95 = allocs[0.95].reindex(w_mm.index).fillna(0.0)
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    l1 = lambda a, b: float(np.abs(a.values - b.values).sum())
    sim = pd.DataFrame({
        "cosine_similarity": [cos(w_mm.values, w_b99.values), cos(w_mm.values, w_b95.values)],
        "L1_weight_distance": [l1(w_mm, w_b99), l1(w_mm, w_b95)],
        "shared_top5": [
            len(set(w_mm.sort_values(ascending=False).head(5).index)
                & set(w_b99.sort_values(ascending=False).head(5).index)),
            len(set(w_mm.sort_values(ascending=False).head(5).index)
                & set(w_b95.sort_values(ascending=False).head(5).index))],
    }, index=["minimax vs beta=0.99", "minimax vs beta=0.95"])
    print("\nMinimax vs beta scenarios (1.0 cosine = identical tilt):")
    print(sim.to_string(float_format=lambda v: f"{v:.3f}"))
    sim.to_csv(os.path.join(OUT_DIR, "part4_minimax_vs_beta.csv"))

    # Figure: minimax vs beta=0.99 vs beta=0.95 top weights
    fig, ax = plt.subplots(figsize=(9, 4.5))
    top_names = w_mm.sort_values(ascending=False).head(8).index
    xpos = np.arange(len(top_names))
    ax.bar(xpos - 0.27, w_mm[top_names] * 100, width=0.27, label="Minimax", color="#d95f0e")
    ax.bar(xpos, w_b99[top_names] * 100, width=0.27, label="beta=0.99", color="#2c7fb8")
    ax.bar(xpos + 0.27, w_b95[top_names] * 100, width=0.27, label="beta=0.95", color="#9ecae1")
    ax.set_xticks(xpos)
    ax.set_xticklabels(top_names, rotation=60, fontsize=8)
    ax.set_ylabel("Weight [%]")
    ax.set_title("Figure 6. Minimax tilts like the high-beta portfolio "
                 f"(cosine {sim.loc['minimax vs beta=0.99','cosine_similarity']:.2f})")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig6_part4_minimax_vs_beta.png")

    # ===================================================================
    # TASK 5 - Monthly re-optimization in test year (rolling 12 months)
    # ===================================================================
    print("\n" + "=" * 60 + "\nPART 5: Monthly Re-optimization (rolling 12m)\n" + "=" * 60)
    rets_all = pd.concat([rets_train, rets_test], axis=0)
    months_all = rets_all.index.to_period("M")
    test_months = sorted(rets_test.index.to_period("M").unique())

    rows, weights_by_month = [], {}
    for mo in test_months:
        # Rolling window = the 12 calendar months immediately preceding mo.
        # (Assignment: Jan-2020 uses Jan-2019..Dec-2019; Feb-2020 uses Feb-2019..Jan-2020.)
        window = rets_all[(months_all >= (mo - 12)) & (months_all <= (mo - 1))]
        if len(window) < 20:
            continue
        sol = solve_cvar_lp(window, beta=BETA, R=RMIN)
        e = evaluate_portfolio(sol["x"], rets_all[months_all == mo], beta=BETA)
        # Static baseline (Task-2 weights) evaluated in the same month, for comparison
        e_static = evaluate_portfolio(sol2["x"], rets_all[months_all == mo], beta=BETA)
        rows.append({"Month": str(mo),
                     "rolling_CVaR": e["CVaR_beta_loss"],
                     "static_CVaR": e_static["CVaR_beta_loss"],
                     "rolling_avg_ret": e["avg_daily_return"],
                     "rolling_vol": e["std_daily_return"]})
        weights_by_month[str(mo)] = pd.Series(sol["x"], index=sol["assets"])

    m5 = pd.DataFrame(rows).set_index("Month")
    stats5 = {"avg_CVaR": m5["rolling_CVaR"].mean(),
              "std_CVaR": m5["rolling_CVaR"].std(ddof=1),
              "min_CVaR": m5["rolling_CVaR"].min(),
              "max_CVaR": m5["rolling_CVaR"].max(),
              "max_month": m5["rolling_CVaR"].idxmax()}
    print(m5.map(lambda v: f"{v:.4f}" if isinstance(v, float) else v).to_string())
    print(f"\nRolling avg CVaR {_pct(stats5['avg_CVaR'])}  std {_pct(stats5['std_CVaR'])}  "
          f"min {_pct(stats5['min_CVaR'])}  max {_pct(stats5['max_CVaR'])} ({stats5['max_month']})")
    months_rolling_better = int((m5["rolling_CVaR"] < m5["static_CVaR"]).sum())
    print(f"Rolling beat the static 2019 portfolio in {months_rolling_better}/{len(m5)} months "
          f"(static avg CVaR {_pct(m5['static_CVaR'].mean())}).")
    m5.to_csv(os.path.join(OUT_DIR, "part5_monthly.csv"))
    results["Task 5 - Avg monthly CVaR (%s)" % test_year] = stats5["avg_CVaR"]
    results["Task 5 - Min monthly CVaR (%s)" % test_year] = stats5["min_CVaR"]
    results["Task 5 - Max monthly CVaR (%s)" % test_year] = stats5["max_CVaR"]

    # Figure: rolling vs static monthly CVaR  [feedback #4 - why rolling isn't always better]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(m5.index, m5["rolling_CVaR"] * 100, "o-", label="Rolling (re-optimized)", color="#2c7fb8")
    ax.plot(m5.index, m5["static_CVaR"] * 100, "s--", label=f"Static {train_year} weights", color="#d95f0e")
    ax.axhline(e_out["CVaR_beta_loss"] * 100, color="#999", ls=":",
               label=f"Static full-year {test_year} CVaR")
    ax.set_ylabel(f"Monthly CVaR (loss), beta={BETA} [%]")
    ax.set_title("Figure 7. Rolling re-optimization tracks the static portfolio closely\n"
                 f"and is not consistently lower (rolling won {months_rolling_better}/{len(m5)} months)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig7_part5_rolling_vs_static.png")

    # Figure: monthly avg return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(m5.index, m5["rolling_avg_ret"] * 100,
           color=["#31a354" if v >= 0 else "#de2d26" for v in m5["rolling_avg_ret"]])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Avg daily return [%]")
    ax.set_title(f"Figure 8. Realized monthly average return of the rolling portfolio ({test_year})")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, "fig8_part5_monthly_return.png")

    # ===================================================================
    # TASK 6 - Stability of monthly allocations
    # ===================================================================
    print("\n" + "=" * 60 + "\nPART 6: Stability\n" + "=" * 60)
    months = list(weights_by_month.keys())
    srows = []
    for i in range(1, len(months)):
        prev, cur = weights_by_month[months[i - 1]], weights_by_month[months[i]]
        delta = (cur - prev).abs()
        srows.append({"from": months[i - 1], "to": months[i],
                      "max_change": float(delta.max()),
                      "mover": delta.idxmax(),
                      "stable": bool(delta.max() <= 0.05)})
    stab = pd.DataFrame(srows)
    share_stable = 100.0 * stab["stable"].mean() if len(stab) else float("nan")
    print(stab.assign(max_change=lambda d: d["max_change"].map(_pct)).to_string(index=False))
    print(f"\nShare of stable (<=5pp) transitions: {share_stable:.2f}%")
    stab.to_csv(os.path.join(OUT_DIR, "part6_stability.csv"), index=False)
    results["Task 6 - Share of stable transitions"] = share_stable / 100.0

    # Figure: max weight change per transition with the 5pp threshold
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xlab = [f"{r['from']}->{r['to'][-2:]}" for _, r in stab.iterrows()]
    ax.bar(xlab, stab["max_change"] * 100,
           color=["#31a354" if s else "#de2d26" for s in stab["stable"]])
    ax.axhline(5, color="k", ls="--", label="5pp stability threshold")
    ax.set_ylabel("Max single-asset weight change [pp]")
    ax.set_title(f"Figure 9. Month-to-month turnover ({share_stable:.0f}% of transitions stable)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    _save(fig, "fig9_part6_stability.png")

    # ===================================================================
    # KEY RESULTS SUMMARY
    # ===================================================================
    print("\n" + "=" * 60 + "\nKEY RESULTS SUMMARY\n" + "=" * 60)
    key = pd.Series({k: _pct(v) for k, v in results.items()}, name="Value")
    print(key.to_string())
    pd.Series(results, name="value").to_csv(os.path.join(OUT_DIR, "key_results.csv"))
    print(f"\nAll tables and figures written to {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
