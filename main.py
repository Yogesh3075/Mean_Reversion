
from __future__ import annotations
import warnings
import os
import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Universe Definitions
# ─────────────────────────────────────────────────────────────────────────────

NIFTY50_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "BHARTIARTL", "INFY", "SBIN", "LICI", "HINDUNILVR", "ITC", "LT", "KOTAKBANK", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", "ADANIENT", "ADANIPORTS", "NTPC", "TITAN", "ONGC", "WIPRO", "ASIANPAINT", "AXISBANK", "ULTRACEMCO", "BAJAJFINSV", "TATASTEEL", "POWERGRID", "NESTLEIND", "JSWSTEEL", "COALINDIA", "M&M", "TECHM", "INDUSINDBK", "HINDALCO", "GRASIM", "BPCL", "CIPLA", "SBILIFE", "DIVISLAB", "APOLLOHOSP", "DRREDDY", "TRENT", "EICHERMOT", "BAJAJ-AUTO", "BRITANNIA", "HEROMOTOCO", "SHRIRAMFIN", "BEL"]
NEXT50_SYMBOLS = ["ABB", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "AUROPHARMA", "BANKBARODA", "BERGEPAINT", "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "DABUR", "DLF", "DMART", "GAIL", "GODREJCP", "HAVELLS", "HDFCAMC", "HINDPETRO", "ICICIGI", "ICICIPRULI", "INDUSTOWER", "IOC", "IRCTC", "JIOFIN", "JSWENERGY", "LODHA", "LUPIN", "MAXHEALTH", "MOTHERSON", "MUTHOOTFIN", "NAUKRI", "NHPC", "NMDC", "OBEROIRLTY", "PAGEIND", "PERSISTENT", "PFC", "PIIND", "PNB", "RECLTD", "SAIL", "SIEMENS", "TORNTPHARM", "TVSMOTOR", "UBL", "UNIONBANK", "VBL", "VEDL"]
ALL_SYMBOLS = NIFTY50_SYMBOLS + NEXT50_SYMBOLS

# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicators (NumPy Optimized)
# ─────────────────────────────────────────────────────────────────────────────

def _sma(s, p): return pd.Series(s).rolling(p).mean().values
def _std(s, p): return pd.Series(s).rolling(p).std().values
def _rsi(s, p=14):
    delta = pd.Series(s).diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0], down[down > 0] = 0, 0
    roll_up = up.rolling(p).mean()
    roll_down = down.abs().rolling(p).mean()
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))

# ─────────────────────────────────────────────────────────────────────────────
# Mean Reversion Strategy Logic
# ─────────────────────────────────────────────────────────────────────────────

class MeanReversionStrategy(Strategy):
    bb_period, bb_std = 20, 2.0
    entry_z, exit_z = 2.0, 0.0
    rsi_period, rsi_oversold, rsi_overbought = 14, 35, 65
    position_size = 0.10

    def init(self):
        close = self.data.Close
        self.ma = self.I(_sma, close, self.bb_period)
        self.sd = self.I(_std, close, self.bb_period)
        self.rsi = self.I(_rsi, close, self.rsi_period)

    def next(self):
        z_score = (self.data.Close[-1] - self.ma[-1]) / self.sd[-1] if self.sd[-1] > 0 else 0

        if self.position:
            if z_score >= self.exit_z or self.rsi[-1] >= self.rsi_overbought:
                self.position.close()
        else:
            if z_score < -self.entry_z and self.rsi[-1] < self.rsi_oversold:
                qty = int((self.equity * self.position_size) // self.data.Close[-1])
                if qty > 0:
                    self.buy(size=qty)

# ─────────────────────────────────────────────────────────────────────────────
# Data Management
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv(symbol: str, data_dir: str = "data/parquet", start: str = "2018-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{symbol}.parquet")

    if not os.path.exists(path):
        df = yf.download(f"{symbol}.NS", start="2017-01-01", end="2026-04-15", progress=False)
        if df.empty: raise FileNotFoundError(f"Ticker {symbol} not found.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df.to_parquet(path, engine='pyarrow')

    df = pd.read_parquet(path, engine='pyarrow')
    df.columns = [str(c).capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return df.sort_index().dropna().loc[start:end]

# ─────────────────────────────────────────────────────────────────────────────
# KPI Diagnostic Suite
# ─────────────────────────────────────────────────────────────────────────────

def _extract_full_metrics(stats, symbol: str) -> dict:
    w_rate = stats['Win Rate [%]'] / 100
    p_factor = stats['Profit Factor'] if not np.isnan(stats['Profit Factor']) and stats['Profit Factor'] != np.inf else 0
    kelly = w_rate - ((1 - w_rate) / p_factor) if p_factor > 0 else 0

    return {
        "Symbol": symbol,
        "Total Return (%)": round(stats['Return [%]'], 2),
        "CAGR (%)": round(stats['Return (Ann.) [%]'], 2),
        "Buy & Hold Return (%)": round(stats['Buy & Hold Return [%]'], 2),
        "Final Equity (INR)": round(stats['Equity Final [$]'], 2),
        "Peak Equity (INR)": round(stats['Equity Peak [$]'], 2),
        "Max Drawdown (%)": round(stats['Max. Drawdown [%]'], 2),
        "Avg Drawdown (%)": round(stats['Avg. Drawdown [%]'], 2),
        "Volatility (Ann. %)": round(stats['Volatility (Ann.) [%]'], 2),
        "Sharpe Ratio": round(stats['Sharpe Ratio'], 4),
        "Sortino Ratio": round(stats['Sortino Ratio'], 4),
        "Calmar Ratio": round(stats['Calmar Ratio'], 4),
        "Total Trades": int(stats['# Trades']),
        "Win Rate (%)": round(stats['Win Rate [%]'], 2),
        "Best Trade (%)": round(stats['Best Trade [%]'], 2),
        "Worst Trade (%)": round(stats['Worst Trade [%]'], 2),
        "Avg Trade (%)": round(stats['Avg. Trade [%]'], 2),
        "Profit Factor": round(p_factor, 2),
        "Expectancy (%)": round(stats['Expectancy [%]'], 2),
        "SQN": round(stats['SQN'], 4) if not np.isnan(stats['SQN']) else 0,
        "Kelly Criterion": round(kelly, 4),
        "Exposure Time (%)": round(stats['Exposure Time [%]'], 2),
        "Avg Trade Duration": str(stats['Avg. Trade Duration']),
        "Max Trade Duration": str(stats['Max. Trade Duration']),
        "Max Drawdown Duration": str(stats['Max. Drawdown Duration']),
        "Avg Drawdown Duration": str(stats['Avg. Drawdown Duration']),
        "_equity_curve": stats['_equity_curve']  # Keep for plotting
    }

# ─────────────────────────────────────────────────────────────────────────────
# Runner & Plotting
# ─────────────────────────────────────────────────────────────────────────────

def run_universe_research(symbols: List[str]):
    os.makedirs("results", exist_ok=True)
    results_list = []

    print(f"🚀 NiftyQuantSignals: Researching {len(symbols)} Stocks...")
    for symbol in tqdm(symbols):
        try:
            df = load_ohlcv(symbol)
            bt = Backtest(df, MeanReversionStrategy, cash=1_000_000, trade_on_close=True)
            stats = bt.run()
            results_list.append(_extract_full_metrics(stats, symbol))
        except Exception:
            continue

    # Save CSV (Drop the equity curve object before saving)
    df_final = pd.DataFrame(results_list).drop(columns=['_equity_curve'])
    df_final.to_csv("results/universe_results.csv", index=False)
    print("\n✅ Institutional Metrics saved to: results/universe_results.csv")

    return results_list

def plot_research_equity_lines(results_list: list):
    plt.figure(figsize=(14, 8))

    # Sort by Sharpe and take top 10 for visualization
    top_10 = sorted(results_list, key=lambda x: x['Sharpe Ratio'], reverse=True)[:10]

    for item in top_10:
        equity = item['_equity_curve']['Equity']
        # Normalize so all lines start at 100
        norm_equity = (equity / equity.iloc[0]) * 100
        plt.plot(norm_equity, label=f"{item['Symbol']} (Sharpe: {item['Sharpe Ratio']})", linewidth=1.5, alpha=0.8)

    plt.title("Mean Reversion Alpha: Normalized Equity Curves (Top 10 Performance)", fontsize=16)
    plt.ylabel("Normalized Equity (Base 100)", fontsize=12)
    plt.xlabel("Timeline", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig("results/mean_reversion_lines.png")
    print("📈 Research Chart saved to: results/mean_reversion_lines.png")
    plt.show()

if __name__ == "__main__":
    # Execute Full Universe Research
    research_data = run_universe_research(ALL_SYMBOLS)

    if research_data:
        plot_research_equity_lines(research_data)