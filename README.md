# QuantVision: An Effective Stock Analysis MAS (Multi-Agent System)
## ETH Oxford '26 Submission - AI Middleware and Application Main Track - Rabot Effectful Programming Sponsor Track
## A "White Box" Financial Intelligence System built with an explicit IO Monad Runtime in Python.

This project implements a hedge-fund-grade stock analysis pipeline where every side effect—from network calls to stochastic simulations—is encapsulated in a custom Effect System. Unlike standard imperative Python scripts, this system separates Pure Domain Logic (Math/Strategy) from Impure Execution (I/O), ensuring deterministic testing and composable workflows. A Streamlit version of the project is also available in the Streamlit_Interface folder, providing an enhanced visual interface. 

# Architecture
To satisfy the "Effectful Programming Core" bounty requirements, this system does not simply "do stuff" (like calling APIs directly inside functions). Instead, it strictly adheres to the IO Monad pattern:

- The IO Container: A custom ```IO[T]``` class wraps all impure actions.

- Pure Core: All financial models (Brownian Motion, Technical Indicators, Valuation) are pure functions that accept data and return results deterministically.

- Impure Shell: Data fetching and model training are wrapped in IO effects.

- Composition: Tools use ```.map()``` and ```.flat_map()``` to build execution pipelines without running them.

- Runtime at the Edge: Effects are only executed via ```.unsafe_run()``` at the very boundary of the agent tools.

# Feature & Agents
The system orchestrates a team of specialized agents using LangGraph:

## The Quant (Trend Agent):

- Brownian Motion Model: Stochastic Monte Carlo simulation (1,000 paths) to estimate Drift and Volatility.

- ML Model: Facebook Prophet (Additive Regression) for time-series forecasting.

## The Value Investor (Fundamental Agent):

- Analyzes P/E Ratios, PEG, Profit Margins, and Debt-to-Equity to determine intrinsic value.

- The Chartist (Technical Agent):

- Calculates RSI (Momentum), MACD (Trend), and Bollinger Bands (Volatility) for entry/exit signals.

## The Risk Manager (Risk Agent):

- Computes Value at Risk (Confidence Level 99%) and Maximum Drawdown to assess downside exposure.

## The Researcher (Noise Agent):

- Uses Valyu API to fetch and filter real-time market news and sentiment.

# Prerequisites

- Python 3.12+
  
- ```VALYU``` and ```OPENAI``` API keys

# Installation

Clone the repo

```bash
git clone https://github.com/your-username/effectful-stock-mas.git
cd effectful-stock-mas
```

Install dependencies

```bash
pip install -U langchain langchain-openai langchain-community langgraph valyu prophet yfinance pandas numpy
```

Set environment variables

```bash
export OPENAI_API_KEY="sk-..."
export VALYU_API_KEY="..."
```


# How to run


Open Jupyter Notebook ```tools_using_effects_final_final.ipynb```

Modify the query in the final cell to test different stocks:

```bash
result = workflow.invoke({
    "query": "Perform a deep dive analysis on NVDA stock"
})
```

Execute all cells

**Note**: This project was created in collaboration with Allen Zhang (Imperial College London) and Burak Klinic (University of Bristol).
