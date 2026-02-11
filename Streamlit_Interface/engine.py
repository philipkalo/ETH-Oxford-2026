import os
import operator
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Annotated, Literal, TypedDict, List
from prophet import Prophet


# --- LIBRARIES ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.prebuilt import create_react_agent 
from pydantic import BaseModel, Field
from valyu import Valyu 
from langchain.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model



model = init_chat_model("gpt-4.1")

from typing import TypeVar, Callable, Generic, Any
from dataclasses import dataclass

T = TypeVar("T")
U = TypeVar("U")

@dataclass
class IO(Generic[T]):
    """
    A pure description of a side-effectful computation.
    Nothing runs until .unsafe_run() is called.
    """
    effect: Callable[[], T]

    @staticmethod
    def pure(value: T) -> "IO[T]":
        """Lift a pure value into the IO context."""
        return IO(lambda: value)

    @staticmethod
    def fail(error: Exception) -> "IO[Any]":
        """Lift an error into the IO context."""
        def _raise(): raise error
        return IO(_raise)

    def map(self, f: Callable[[T], U]) -> "IO[U]":
        """Apply a pure function to the result of the effect."""
        return IO(lambda: f(self.effect()))

    def flat_map(self, f: Callable[[T], "IO[U]"]) -> "IO[U]":
        """Chain a new effect based on the result of the previous one."""
        return IO(lambda: f(self.effect()).unsafe_run())

    def attempt(self) -> "IO[T | Exception]":
        """Materialize errors into values (Better failure handling)."""
        def _safe_run():
            try:
                return self.effect()
            except Exception as e:
                return e
        return IO(_safe_run)

    def unsafe_run(self) -> T:
        """The 'Edge' - actually executes the side effects."""
        return self.effect()

# Helper for composing multiple IOs
def sequence(ios: list[IO[T]]) -> IO[list[T]]:
    def _run_all():
        return [io.unsafe_run() for io in ios]
    return IO(_run_all)


from typing import Annotated, Literal, TypedDict
import operator


class AgentInput(TypedDict):
    """Simple input state for each subagent."""
    query: str


class AgentOutput(TypedDict):
    """Output from each subagent."""
    source: str
    result: str


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""
    source: Literal["quant", "research"]
    query: str


class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]  
    final_answer: str

class BrownianParams(TypedDict):
    mu: float
    sigma: float
    last_price: float
    annual_vol: float
    annual_drift: float


# --- EFFECT DEFINITIONS (I/O Boundary) ---

def fetch_stock_history_io(ticker: str, years: int = 2) -> IO[pd.DataFrame]:
    """Effect: Network Call to Yahoo Finance."""
    def _fetch():
        end_date = pd.Timestamp.today().normalize()
        start_date = end_date - pd.DateOffset(years=years)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
            if isinstance(data, pd.DataFrame) and ticker in data.columns:
                 data = data[ticker]
        elif 'Close' in data.columns:
            data = data['Close']
        if isinstance(data, pd.DataFrame):
             data = data.iloc[:, 0]
        return data
    return IO(_fetch)

def run_monte_carlo_io(params: BrownianParams, days: int = 30, scenarios: int = 1000) -> IO[pd.DataFrame]:
    """Effect: Random Number Generation & Simulation."""
    def _sim():
        mu, sigma, S0 = params['mu'], params['sigma'], params['last_price']
        dt = 1
        returns = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=(days, scenarios))
        price_paths = np.vstack([np.full((1, scenarios), S0), S0 * np.exp(np.cumsum(returns, axis=0))])
        return pd.DataFrame(price_paths)
    return IO(_sim)

def valyu_search_io(query: str) -> IO[dict]:
    """
    Effect: External API Search with Strict Relevance Filters.
    Returns a Dictionary (JSON), not a string, to avoid premature formatting.
    """
    def _search():
        try:
            client = Valyu(api_key=os.environ.get("VALYU_API_KEY"))
            
            # API-LEVEL FILTERING 
            return client.search(
                query=query,
                max_num_results=3, 
                response_length="short"  
            )
            
        except Exception as e:
            return {"error": str(e), "results": []}
            
    return IO(_search)

def prophet_predict_io(df: pd.DataFrame, days: int = 30) -> IO[pd.DataFrame]:
    """Effect: Heavy Computation / Model Training."""
    def _train_and_predict():
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        return forecast
    return IO(_train_and_predict)


def fetch_fundamentals_io(ticker: str) -> IO[dict]:
    """Effect: Fetch fundamental metadata (P/E, Margins, Debt) from YFinance."""
    def _fetch():
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.info
    return IO(_fetch)


def fetch_stock_data_io(ticker: str) -> IO[pd.DataFrame]:
    """
    Effect wrapper to fetch stock data for 2 years.
    Returns an IO-like object with the data.
    """
    def _fetch():
        data = yf.download(ticker, period="2y", interval="1d")
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return data
    return io.IO(_fetch)


# --- PURE DOMAIN TYPES & LOGIC ---

class BrownianParams(TypedDict):
    mu: float
    sigma: float
    last_price: float
    annual_vol: float
    annual_drift: float

def calculate_brownian_params_pure(prices: pd.Series) -> BrownianParams:
    """Pure: Extract statistical parameters from data."""
    if len(prices) < 2:
        raise ValueError("Not enough data")

    daily_returns = ((prices / prices.shift(1)) - 1).dropna()
    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)
    last_price = float(prices.iloc[-1])
    
    return {
        "mu": mu,
        "sigma": sigma,
        "last_price": last_price,
        "annual_vol": sigma * np.sqrt(252),
        "annual_drift": mu * 252
    }

def format_brownian_output_pure(sim_df: pd.DataFrame, ticker: str, params: BrownianParams) -> str:
    """Pure: Format the simulation results into a detailed table."""
    days = sim_df.shape[0]
    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=days, freq='B')
    
    stats_df = pd.DataFrame({
        'Date': future_dates,
        'Mean': sim_df.mean(axis=1),
        'Low (5%)': sim_df.quantile(0.05, axis=1),
        'High (95%)': sim_df.quantile(0.95, axis=1)
    })
    
    display_df = stats_df.iloc[::5].copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    table_str = display_df.to_string(index=False, float_format="%.2f")
    
    return (f"Brownian Motion Analysis for {ticker}:\n"
            f"--- TECHNICAL PARAMETERS ---\n"
            f"Annualized Volatility: {params['annual_vol']:.2%}\n"
            f"Annualized Drift: {params['annual_drift']:.2%}\n"
            f"--- FORECAST TABLE (Weekly Snapshots) ---\n"
            f"```text\n{table_str}\n```")  

def prepare_prophet_data_pure(data: pd.DataFrame) -> pd.DataFrame:
    """Pure Logic: Rename columns for Prophet."""
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()
        
    df = df.reset_index()
    
    date_col_name = df.columns[0]
    price_col_name = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    df['ds'] = pd.to_datetime(df[date_col_name]).dt.tz_localize(None)
    df['y'] = pd.to_numeric(df[price_col_name], errors='coerce')
    df = df.dropna(subset=['y'])

    return df[['ds', 'y']]

def format_prophet_output(forecast: pd.DataFrame, ticker: str) -> str:
    """Pure transformation of Prophet results to text with a table."""
    future_data = forecast.tail(30)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    latest_pred = forecast.iloc[-1]['yhat']
    trend = "UP" if latest_pred > forecast.iloc[0]['yhat'] else "DOWN"
    
    future_data.columns = ['Date', 'Target', 'Low', 'High']
    display_df = future_data.iloc[::5].copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    table_str = display_df.to_string(index=False, float_format="%.2f")

    return (f"ML Analysis for {ticker}\n"
            f"Trend: {trend}\n"
            f"--- FORECAST TABLE (Next 30 Days) ---\n"
            f"```text\n{table_str}\n```") 


def format_search_results_pure(response: dict) -> str:
    """Pure Logic: Convert structured API JSON into report."""
    if "error" in response and response["error"]:
        return f"Search Error: {response['error']}"
    
    results = (
        response.get("results") or 
        response.get("contents") or 
        response.get("data") or 
        response.get("hits") or 
        []
    )
    
    if isinstance(results, str):
        return f"Raw Search Output: {results[:1000]}..."
        
    if not isinstance(results, list):
        return f"Unexpected API response format: {type(results)}"

    if not results:
        keys_found = list(response.keys()) if isinstance(response, dict) else "Not a dict"
        return f"No relevant news found. (Debug info: {keys_found})"
    
    formatted = ["### Market Research Summary"]
    
    for item in results[:5]:
        if isinstance(item, dict):
            title = item.get("title", "Untitled")
            source = item.get("source_domain", "Unknown Source")
            url = item.get("url", "#")
            content = item.get("content", "")[:300] 
        else:
            title = getattr(item, "title", "Untitled")
            source = getattr(item, "source_domain", "Unknown Source")
            url = getattr(item, "url", "#")
            content = getattr(item, "content", "")[:300]

        formatted.append(f"- **{title}** ({source})\n  *\"{content}...\"*\n  [Link]({url})")
        
    final_str = "\n\n".join(formatted)
    
    if len(final_str) > 2000:
        return final_str[:2000] + "\n... [TRUNCATED FOR RATE LIMIT SAFETY]"
    
    return final_str

def calculate_technicals_pure(prices: pd.Series) -> dict:
    """
    Pure: Calculates RSI, MACD, and Bollinger Bands from price series.
    Returns a dictionary of the latest values and interpreting signals.
    """
    if len(prices) < 30:
        return {"error": "Not enough data for technical analysis (need >30 days)"}
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    sma20 = prices.rolling(window=20).mean()
    std20 = prices.rolling(window=20).std()
    upper_band = sma20 + (std20 * 2)
    lower_band = sma20 - (std20 * 2)
    
    latest_price = prices.iloc[-1]
    latest_rsi = rsi.iloc[-1]
    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    latest_upper = upper_band.iloc[-1]
    latest_lower = lower_band.iloc[-1]
    
    rsi_signal = "NEUTRAL"
    if latest_rsi > 70: rsi_signal = "OVERBOUGHT (Sell)"
    elif latest_rsi < 30: rsi_signal = "OVERSOLD (Buy)"
    
    macd_signal = "BULLISH" if latest_macd > latest_signal else "BEARISH"
    
    bb_signal = "NEUTRAL"
    if latest_price > latest_upper: bb_signal = "UPPER BREAKOUT (Sell)"
    elif latest_price < latest_lower: bb_signal = "LOWER BREAKOUT (Buy)"
    
    return {
        "price": latest_price,
        "rsi": latest_rsi,
        "rsi_signal": rsi_signal,
        "macd": latest_macd,
        "macd_signal": macd_signal,
        "bb_upper": latest_upper,
        "bb_lower": latest_lower,
        "bb_signal": bb_signal
    }

def format_technicals_pure(data: dict, ticker: str) -> str:
    """Pure: Formats technical analysis results into a readable report."""
    if "error" in data:
        return f"Technical Analysis Failed: {data['error']}"
        
    return (f"Technical Analysis (Chartist) for {ticker}\\n"
            f"--- INDICATORS ---\\n"
            f"Price: ${data['price']:.2f}\\n"
            f"RSI (14): {data['rsi']:.2f} -> {data['rsi_signal']}\\n"
            f"MACD: {data['macd']:.4f} (Signal: {data['macd_signal']})\\n"
            f"Bollinger Bands: ${data['bb_lower']:.2f} - ${data['bb_upper']:.2f}\\n"
            f"Band Status: {data['bb_signal']}\\n"
            f"------------------")


def analyze_fundamentals_pure(info: dict, ticker: str) -> str:
    """Pure: Analyzes financial health and valuation metrics."""
    if not info:
        return f"Fundamental Analysis Error: No data found for {ticker}."
    
    current_price = info.get('currentPrice', 0)
    target_mean = info.get('targetMeanPrice', 0)
    
    pe_ratio = info.get('trailingPE', None)
    forward_pe = info.get('forwardPE', None)
    peg_ratio = info.get('pegRatio', None)
    pb_ratio = info.get('priceToBook', None)
    
    profit_margin = info.get('profitMargins', 0)
    operating_margin = info.get('operatingMargins', 0)
    roe = info.get('returnOnEquity', 0)
    
    debt_to_equity = info.get('debtToEquity', None)
    free_cash_flow = info.get('freeCashflow', None)
    
    valuation_verdict = "NEUTRAL"
    if peg_ratio and peg_ratio < 1.0: 
        valuation_verdict = "UNDERVALUED (PEG < 1)"
    elif peg_ratio and peg_ratio > 2.0:
        valuation_verdict = "OVERVALUED (PEG > 2)"
        
    health_verdict = "STABLE"
    if debt_to_equity and debt_to_equity > 200:
        health_verdict = "HIGH DEBT RISK"
        
    upside_potential = 0.0
    if current_price and target_mean:
        upside_potential = ((target_mean - current_price) / current_price) * 100

    fcf_str = f"${free_cash_flow/1e9:.2f}B" if free_cash_flow else "N/A"
    
    return (f"Fundamental Analysis (Value Investor) for {ticker}\\n"
            f"--- VALUATION ---\\n"
            f"P/E Ratio: {pe_ratio} (Forward: {forward_pe})\\n"
            f"PEG Ratio: {peg_ratio} -> {valuation_verdict}\\n"
            f"Price-to-Book: {pb_ratio}\\n"
            f"Analyst Target Upside: {upside_potential:.2f}%\\n"
            f"--- PROFITABILITY & HEALTH ---\\n"
            f"Profit Margin: {profit_margin:.2%}\\n"
            f"ROE: {roe:.2%}\\n"
            f"Debt-to-Equity: {debt_to_equity}\\n"
            f"Free Cash Flow: {fcf_str}\\n"
            f"Health Status: {health_verdict}")



def analyze_risks_pure(data, ticker: str) -> str:
    """
    Pure computation of Max Drawdown and Daily VaR.
    """
    close = data['Close']
    
    # Maximum Drawdown
    cumulative_max = close.cummax()
    drawdown = (close - cumulative_max) / cumulative_max
    max_dd = round(drawdown.min() * 100, 2)
    
    # Value at Risk (99% confidence)
    daily_returns = close.pct_change().dropna()
    var = round(daily_returns.quantile(0.01) * 100, 2)
    
    return (
        f"Risk Metrics for {ticker} over 2 years:\n"
        f"- Maximum Drawdown: {max_dd}%\n"
        f"- Daily Value at Risk (99%): {var}%"
    )




@tool
def brownianModel(TICKER: str):
    """
    Uses an Effect System to model stock prediction.
    """
    program = (
        fetch_stock_history_io(TICKER)
        .map(calculate_brownian_params_pure)
        .flat_map(lambda params: 
            run_monte_carlo_io(params).map(
                lambda sim_df: format_brownian_output_pure(sim_df, TICKER, params)
            )
        )
    )

    result = program.attempt().unsafe_run()
    
    if isinstance(result, Exception):
        return f"Brownian Model Failed: {str(result)}"
    return result

@tool
def mlModel(ticker: str):
    """ 
    Uses an Effect System to model Facebook Prophet predictions.
    """
    program = (
        fetch_stock_history_io(ticker)
        .map(prepare_prophet_data_pure)
        .flat_map(lambda df: prophet_predict_io(df))
        .map(lambda forecast: format_prophet_output(forecast, ticker))
    )

    result = program.attempt().unsafe_run()
    
    if isinstance(result, Exception):
        return f"ML Model Failed: {str(result)}"
    return result

@tool
def valyu_search_tool(query: str):
    """
    Effectful search wrapper with Relevance Filtering.
    """
    program = (
        valyu_search_io(query)
        .map(format_search_results_pure)
    )
    
    return program.attempt().unsafe_run()

@tool
def technicalAnalysisModel(TICKER: str):
    """
    Analyzes Technical Indicators (RSI, MACD, Bollinger Bands) to generate buy/sell signals.
    """
    program = (
        fetch_stock_history_io(TICKER)
        .map(calculate_technicals_pure)
        .map(lambda res: format_technicals_pure(res, TICKER))
    )
    
    result = program.attempt().unsafe_run()
    
    if isinstance(result, Exception):
        return f"Technical Analysis Failed: {str(result)}"
    return result

@tool
def fundamentalModel(TICKER: str):
    """
    Analyzes Fundamental Ratios (P/E, PEG, Debt, Margins) to determine intrinsic value.
    """
    program = (
        fetch_fundamentals_io(TICKER)
        .map(lambda info: analyze_fundamentals_pure(info, TICKER))
    )
    
    result = program.attempt().unsafe_run()
    
    if isinstance(result, Exception):
        return f"Fundamental Analysis Failed: {str(result)}"
    return result


@tool
def riskModel(TICKER: str):
    """
    Analyzes stock risk metrics (Max Drawdown and Daily VaR) to assess downside risk.
    """
    # Effect: fetch stock data
    program = (
        fetch_stock_data_io(TICKER)
        # Pure transformation: calculate risk metrics
        .map(lambda data: analyze_risks_pure(data, TICKER))
    )
    
    # Execute the program
    result = program.attempt().unsafe_run()
    
    if isinstance(result, Exception):
        return f"Risk Analysis Failed: {str(result)}"
    return result


trend_prompt = (
    "You are a Quantitative Analyst. Use the provided ML, Statistical, Technical, and Fundamental tools to analyze the stock ticker provided. "
    "ONLY ENTER THE ABBREVIATION OF THE STOCK TO THE TOOLS. "
    "Your report must be detailed and data-heavy. You MUST include:\n"
    "1. The exact current price of the stock.\n"
    "2. The specific daily price targets for the next 30 days from the ML (Prophet) model.\n"
    "3. The median prediction and confidence intervals from the Brownian motion model.\n"
    "4. Technical Analysis: RSI, MACD, Bollinger Bands signals.\n"
    "5. Fundamental Analysis: P/E, PEG Ratio, Profit Margins, and Fair Value verdicts.\n" 
    "6. A clear statement of the trend direction (UP/DOWN/FLAT) based on the consensus of all models.\n"
    "7. If a tool fails, explicitly state why (e.g., 'Not enough data')."
)

trend_agent = create_agent(
    model, 
    system_prompt=SystemMessage(content=[{"type": "text", "text": trend_prompt}, {"type": "text", "text": "stock markets"}]), 
    tools=[mlModel, brownianModel, technicalAnalysisModel, fundamentalModel,riskModel] 
)


noise_prompt = (
    "You are a Market Researcher. Use the search tool to find recent news, sentiment, and macro factors affecting the stock. "
    "Do not just summarize; provide a detailed list of findings. You MUST include:\n"
    "1. Specific headlines, dates, and sources of the news you found.\n"
    "2. Direct quotes or key statistics from the search results.\n"
    "3. Any upcoming events (earnings dates, product launches).\n"
    "4. The overall market sentiment supported by specific evidence."
)
noise_agent = create_agent(model, [valyu_search_tool], system_prompt=SystemMessage(content=[{"type": "text", "text": noise_prompt}, {"type": "text", "text": "stock markets"}], ))


from pydantic import BaseModel, Field
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

class ClassificationResult(BaseModel):  
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )

def classify_query(state: RouterState) -> dict:
    """Classify query and spawn agents for BOTH quant and research."""
    structured_llm = model.with_structured_output(ClassificationResult)  

    system_prompt = """You are a Supervisor Agent. 
    When the user asks for a stock prediction, you MUST generate TWO separate instructions:
    
    1. One for the 'quant' agent to run the mathematical models (Brownian & Prophet).
    2. One for the 'research' agent to find news and sentiment.
    
    OUTPUT format:
    Return a list of TWO classifications.
    - Classification 1: source='quant', query='[Ticker Symbol]' (e.g., 'AMZN')
    - Classification 2: source='research', query='[Ticker Symbol] news and sentiment'
    """

    result = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["query"]}
    ])

    return {"classifications": result.classifications}

def route_to_agents(state: RouterState) -> list[Send]:
    """Fan out to agents based on classifications."""
    return [
        Send(c["source"], {"query": c["query"]})  
        for c in state["classifications"]
    ]


def run_trend_agent(state: RouterState):
    """Invokes the Quant Agent"""
    print("Executing Trend Agent")
    response = trend_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    
    return {"results": [{"source": "quant", "result": response["messages"][-1].content}]}

def run_noise_agent(state: RouterState):
    """Invokes the Research Agent"""
    print("Executing Noise Agent")
    response = noise_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    
    return {"results": [{"source": "research", "result": response["messages"][-1].content}]}

def synthesize_results(state: RouterState) -> dict:
    """Combine results from all agents into a comprehensive report."""
    if not state["results"]:
        return {"final_answer": "No results found from any knowledge source."}

    formatted = [
        f"--- REPORT FROM {r['source'].upper()} DEPARTMENT ---\n{r['result']}\n------------------------------------------------"
        for r in state["results"]
    ]

    synthesis_prompt = f"""You are a Senior Investment Analyst compiling a comprehensive Due Diligence Report.
    The user asked: "{state['query']}"

    Your goal is to provide a "White Box" analysis—explaining NOT just the prediction, but HOW the math worked.

    STRICTLY FOLLOW THIS REPORT STRUCTURE:

    1. **Executive Summary**
       - A high-level verdict (Buy/Sell/Hold/Wait).
       - specific mention of whether the stock is "Undervalued" or "Overvalued" based on fundamentals.

    2. **Methodology & Technical Deep-Dive**
       - **Brownian Motion:** State the "Annualized Volatility" and "Drift" percentages.
       - **Fundamentals:** Highlight the Valuation Method (P/E, PEG, Fair Value).
       - **Technicals:** Mention the indicators used (RSI, MACD, Bollinger Bands).

    3. **Quantitative Analysis (The Numbers)**
       - **CRITICAL:** The tools provided DATA TABLES (text spreadsheets) wrapped in code blocks.
       - You **MUST COPY THESE TABLES EXACTLY** into your report. 
       - **DO NOT** convert the tables into bullet points. 
       - Just copy the Markdown code blocks containing the tables.

    4. **Fundamental & Technical Health**
       - **Valuation:** Report P/E, PEG Ratio, and Profit Margins. Is it cheap?
       - **Momentum:** Report RSI (Overbought/Oversold) and MACD signals.
       - **Health:** Debt-to-Equity and Cash Flow status.

    5. **Market Context (The News)**
       - Summarize the news headlines and sentiment.
       - CITE SOURCES.

    6. **Risk Factors & Conclusion**
       - Specific risks (e.g., "High volatility of X% increases downside risk").
       - Final recommendation based on the consensus of Math (Quant), Value (Fundamentals), and Momentum (Technicals).

    Do not shorten the content. USE THE TECHNICAL PARAMETERS (Sigma, Mu, CI) PROVIDED IN THE TEXT."""

    synthesis_response = model.invoke([
        {"role": "system", "content": synthesis_prompt},
        {"role": "user", "content": "\n\n".join(formatted)}
    ])

    return {"final_answer": synthesis_response.content}

workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("quant", run_trend_agent)
    .add_node("research", run_noise_agent)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_to_agents, ["quant", "research"])
    .add_edge("quant", "synthesize")
    .add_edge("research", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)


result = workflow.invoke({
    "query": "can you make predictions on Amazon stock?"
})

print("Original query:", result["query"])
print("\nClassifications:")
for c in result["classifications"]:
    print(f"  {c['source']}: {c['query']}")
print("\n" + "=" * 60 + "\n")
print("Final Answer:")
print(result["final_answer"])








