import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check ticker/start/end.")
    df = df.rename(columns=str.lower)
    # expect columns: open high low close volume
    return df


def ma_crossover_signals(close: pd.Series, fast: int, slow: int) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError("fast MA must be < slow MA")

    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out["ma_fast"] = close.rolling(fast).mean()
    out["ma_slow"] = close.rolling(slow).mean()

    # signal: 1 when fast > slow, 0 otherwise
    out["signal"] = (out["ma_fast"] > out["ma_slow"]).astype(int)

    # position: take today's signal as tomorrow's position (avoid look-ahead)
    out["position"] = out["signal"].shift(1).fillna(0).astype(int)

    return out


def backtest(df_sig: pd.DataFrame, fee_bps: float = 0.0) -> pd.DataFrame:
    out = df_sig.copy()

    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["strategy_gross"] = out["position"] * out["ret"]

    # transaction cost: pay when position changes
    # fee_bps: e.g. 10 means 0.10% per trade
    fee = fee_bps / 10000.0
    trades = out["position"].diff().abs().fillna(0.0)  # 1 when enter/exit
    out["cost"] = trades * fee

    out["strategy_net"] = out["strategy_gross"] - out["cost"]

    out["equity_bh"] = (1.0 + out["ret"]).cumprod()
    out["equity_strat"] = (1.0 + out["strategy_net"]).cumprod()
    return out


def summarize(result: pd.DataFrame) -> dict:
    def cagr(equity: pd.Series) -> float:
        if equity.empty:
            return np.nan
        days = (equity.index[-1] - equity.index[0]).days
        if days <= 0:
            return np.nan
        years = days / 365.25
        return equity.iloc[-1] ** (1 / years) - 1

    def max_dd(equity: pd.Series) -> float:
        peak = equity.cummax()
        dd = equity / peak - 1.0
        return dd.min()

    stats = {
        "start": str(result.index.min().date()),
        "end": str(result.index.max().date()),
        "bh_total_return": float(result["equity_bh"].iloc[-1] - 1.0),
        "strat_total_return": float(result["equity_strat"].iloc[-1] - 1.0),
        "bh_cagr": float(cagr(result["equity_bh"])),
        "strat_cagr": float(cagr(result["equity_strat"])),
        "bh_max_drawdown": float(max_dd(result["equity_bh"])),
        "strat_max_drawdown": float(max_dd(result["equity_strat"])),
        "trades": int(result["position"].diff().abs().fillna(0.0).sum()),
    }
    return stats


def plot(result: pd.DataFrame, ticker: str, fast: int, slow: int, outpath: str | None):
    plt.figure()
    plt.plot(result.index, result["equity_bh"], label="Buy & Hold")
    plt.plot(result.index, result["equity_strat"], label=f"MA({fast},{slow})")
    plt.title(f"{ticker} - MA Crossover Backtest")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.legend()
    plt.tight_layout()

    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=160)
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="SPY")
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default="2025-01-01")
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=60)
    p.add_argument("--fee_bps", type=float, default=0.0)
    p.add_argument("--save_plot", action="store_true")
    args = p.parse_args()

    df = load_data(args.ticker, args.start, args.end)
    sig = ma_crossover_signals(df["close"], args.fast, args.slow)
    res = backtest(sig, fee_bps=args.fee_bps)

    stats = summarize(res)
    print("\n=== Summary ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:>18}: {v:.4f}")
        else:
            print(f"{k:>18}: {v}")

    plot_path = None
    if args.save_plot:
        plot_path = f"results/{args.ticker}_ma_{args.fast}_{args.slow}.png"
    plot(res, args.ticker, args.fast, args.slow, plot_path)


if __name__ == "__main__":
    main()