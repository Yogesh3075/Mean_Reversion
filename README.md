# Mean Reversion


The strategy operates on the principle of **Stationarity**—the assumption that while stock prices trend over the long term, short-term deviations from the 20-day mean are often temporary.

* **Alpha Signal:** A price move exceeding **-2 Standard Deviations** from the 20-day Simple Moving Average (SMA).
* **Confluence Filter:** Relative Strength Index (RSI) < 35 to ensure selling exhaustion.
* **Execution Logic:** Trade-on-Close (signals generated at EOD, executed at next-day Open) to account for realistic market slippage.

---

### Project Architecture
The project is structured into four modular layers:

* **Data Layer (`load_ohlcv`):**
    * Automated ingestion from local Parquet stores or `yfinance` fallback.
    * Dynamic column flattening for MultiIndex data handling.
    * Integration of **India VIX** for volatility-regime filtering.
* **Alpha Engine (`MeanReversionStrategy`):**
    * Built on the `backtesting.py` framework.
    * Vectorized technical indicator calculation using **NumPy** for high-speed batch processing.
* **Diagnostic Suite (`_extract_full_metrics`):**
    * Generates a **25-KPI institutional report**.
    * Includes advanced metrics: Sharpe, Sortino, SQN, and Kelly Criterion.
* **Visualization Layer:**
    * Generates cross-sectional equity curve line graphs.
    * Allows comparison of **"Alpha Velocity"** across the entire universe.

---

### Getting Started

#### **Prerequisites**
* Ensure you have the following Python libraries installed in your virtual environment:
    ```bash
    pip install pandas numpy yfinance backtesting tqdm matplotlib seaborn pyarrow
    ```

#### **Execution**
* Run the full universe backtest and generate the research chart:
    ```bash
    python main.py
    ```

#### **File Outputs**
* **`data/parquet/`**: Local storage of historical stock data.
* **`results/universe_results.csv`**: A comprehensive CSV containing 25 KPIs for every stock in the Nifty 100.
---

### Performance Metrics Explained

* **Sharpe Ratio:** Risk-adjusted return; measures the consistency of the mean reversion edge.
* **SQN (System Quality Number):** Statistical significance; quantifies the likelihood that the results are not due to chance.
* **Max Drawdown:** Peak-to-trough decline; essential for understanding the capital risk during market "tail events."
* **Kelly Criterion:** Theoretical optimal position sizing based on win/loss ratios.

---
