# QuantVision: A Langchain Multi-Agent Stock Analysis Engine



**QuantVision** is a sophisticated financial analysis platform that leverages a multi-agent **LangGraph** workflow to perform automated due diligence on stocks. By combining machine learning, statistical simulations, and real-time market research, it provides a detailed analysis, explaining not just the prediction but the mathematical methodology behind it. This folder, in particular, presents a way to use Streamlit for a cleaner interface. 



## Key Features


* **Multi-Agent Orchestration**: Uses a supervisor agent to "fan out" tasks to specialized Quant and Research departments.

* **Monadic Effect System**: Implements a custom `IO` Monad in Python to manage side effects (API calls, simulations), ensuring high reliability and pure functional logic where possible.

* **Advanced Analytics**:

    * **ML Forecasting**: Utilizes Facebook Prophet for 30-day price trend predictions.

    * **Statistical Modeling**: Runs Monte Carlo simulations via Geometric Brownian Motion.

    * **Technical Analysis**: Computes RSI, MACD, and Bollinger Bands.

    * **Fundamental Metrics**: Analyzes P/E ratios, PEG ratios, and Debt-to-Equity ratios.

* **Real-time Research**: Integrates the **Valyu Search API** for a deep dive into news gathering and sentiment analysis.

* **Storage System**: Stores the most recent 10 session reports for future review and comparison. 



---



## Installation & Setup



### 1. Prerequisites

Ensure you have Python 3.12+ installed on your system.



### 2. Install Dependencies

Navigate to your project directory and install the required packages:



```bash

pip install streamlit langchain-openai langchain-google-genai langgraph yfinance prophet pandas numpy matplotlib valyu python-dotenv
```


### 3. Environment Configuration



Create a `.env` file in the root directory of your project and add your API credentials as shown below:



```env

OPENAI_API_KEY=your_openai_key_here

VALYU_API_KEY=your_valyu_search_key_here

```

## Usage



### Running the Web Interface

Launch the Streamlit dashboard on localhost by executing the following command in your terminal:



```bash

streamlit run app.py
```


***

