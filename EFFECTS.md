# Core Paradigm 

This project implements a Functional Core, Imperative Shell architecture using a custom IO Monad in Python. 

* **Business Logic**: Expressed as pure, referentially transparent pipelines.
* **Side Effects**: Encapsulated in ```IO``` containers and only executed at the "edge of the system (the Tool Runtime)
* **Agent Interaction**: The AI Agent does not execute code directly; it triggers the construction of a pipeline, which is then safely interpreted.

## 1) I/O Boundary (Effect Inventory)

The system observes and interacts with the outside world strictly through these defined effects: 

1) Network I/O:
   * **Market Data**: Fetching historical OHLCV stock data via Yahoo Finance API.
   * **Fundamentals**: Fetching real-time financial data (P/E, Margins) via Yahoo Finance API.
   * **Intelligence**: Fetching semantic search results and news via the Valyu API.
2) File System (Persistence):
   * **Ledger**: Appending structured prediction records to a local append-only log (```prediction_ledger.json```).
   * **Visualization**: Rendering and saving .png charts to the local disk.
3) Randomness & Compute:
   * **Simulation**: Generating stochastic paths for Brownian Motion (Monte Carlo).
   * **Model Training**: Fitting Facebook Prophet models (resource-intensive compute).
  
## 2) Effect Definitions (Code References)

The effect system is built on a custom ```IO[T]``` class defined directly in the main notebook. All effects return an ```IO``` instance, describing an action without performing it. 

**The Runtime Primitive**:

 * ```class IO[T]```: A monadic container with ```.map```, ```.flat_map```, and ```.unsafe_run```.

**The Effect Catalogue (Notebook Internal):**

 * ```fetch_stock_history_io(ticker)```: Wraps ```yfinance.download```.
 * ```fetch_fundamentals_io(ticker)```: Wraps ```yfinance.Ticker.info```
 * ```valyu_search_io(query)```: Wraps ```client.search()``` (returns raw Dict).
 * ```run_monte_carlo_io(params)```: Wraps ```numpy.random``` generation
 * ```prophet_predict_io(df)```: Wraps ```Prophet.fit()``` and ```predict()```.


## 3) Pure Core (Business Logic)

The "Brain" of the application is a set of pure functions defined in the notebook. These functions are deterministic and testable without mocks. 

**Data Flow**: 

1) **Input**: Raw data (DataFrames/Dicts) from effects.
2) **Transformation (Pure)**:
   *```calculate_technicals_pure```: Computes RSI, MACD, and Bollinger Bands from price series.
   *```analyze_fundamentals_pure```: Evaluates valuation (P/E, PEG) and financial health.
   *```analyze_risks_pure```: Calculates Max Drawdown and Value at Risk (VaR)
   * ```calculate_brownian_params_pure```: Derives Drift ($\mu$) and Volatility ($\sigma$) from price history.
   * ```prepare_prophet_data_pure```: Normalizes timestamps for ML.
4) **Output Generation (Pure)**:
   * ```format_search_results_pure```: Formats raw API JSON into a readable Markdown summary (with safety truncation)
   * ```format_brownian_output_pure```: Formats simulation arrays into text tables and confidence intervals.
  
**Execution Path:**: Pipelines compose these steps using ```flat_map```.
   * *Example*: ```build_technical_pipeline``` = ```fetch_history``` $\rightarrow$ ```calculate_technicals``` $\rightarrow$ ```format_output```.

## 4) Runtime

* **Language**: Python 3.12+
* **Effect System**: Custom **IO Monad** implementation (reference: ```effects_core.py```))
   * Python lacks built-in monadic styles like Scala's ```ZIO``` or Haskell's ```IO```. We implemented a lightweight container to enforce referential transparency directly in the script.
* **Execution Style: Interpreter Pattern at the Edge**:
   * The LangChain Tools (```@tool```) act as the interpreter boundary.
   * Inside the tool, ```program = build_pipeline(...)``` constructs the execution plan (pure).
   * ```program.attempt().unsafe_run()``` is the **only** line where side effects are authorized to run. 
