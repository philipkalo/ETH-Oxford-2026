import streamlit as st
from dotenv import load_dotenv
import os
from datetime import datetime


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Set it in .env or environment variables.")

# Import AFTER env is loaded
from engine import workflow
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4.1",
    openai_api_key=api_key
)


if "report_history" not in st.session_state:
    st.session_state.report_history = []

if "selected_report" not in st.session_state:
    st.session_state.selected_report = None


st.set_page_config(page_title="QuantVision", page_icon="📈", layout="wide")


with st.sidebar:
    st.title("⚙️ Model Settings")

    ticker = st.text_input("Stock Ticker", value="AMZN").upper()

    st.info("This system uses a Monadic Effect System (IO Monad) to manage side effects.")

    st.divider()
    st.markdown("### 📚 Previous Reports")

    history = st.session_state.report_history

    if not history:
        st.caption("No reports yet.")
    else:
        for i, item in enumerate(history):
            ts = item["timestamp"].strftime("%Y-%m-%d %H:%M")
            label = f"{item['ticker']} — {ts}"

            if st.button(label, key=f"history_{i}"):
                st.session_state.selected_report = item


st.title("📈 QuantVision")
st.subheader("Multi-Agent Stock Due Diligence Engine")


if st.button("Generate Comprehensive Report", type="primary"):
    with st.status("🚀 Agents working...", expanded=True) as status:
        st.write("🔍 Running Multi-Agent Workflow...")
        result = workflow.invoke({"query": f"Analyze {ticker} stock"})
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    final_report = result.get("final_answer", "No report generated.")

    report_entry = {
        "ticker": ticker,
        "timestamp": datetime.now(),
        "final_report": final_report,
        "raw_results": result.get("results", [])
    }


    st.session_state.report_history.insert(0, report_entry)


    MAX_REPORTS = 10
    st.session_state.report_history = st.session_state.report_history[:MAX_REPORTS]


    st.session_state.selected_report = report_entry


report_to_show = st.session_state.selected_report

if report_to_show:
    st.divider()
    st.markdown(f"## 📄 Report for {report_to_show['ticker']}")
    st.caption(f"Generated at {report_to_show['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    tab1, tab2 = st.tabs(["📄 Final Report", "📊 Technical Logs"])

    with tab1:
        st.markdown(report_to_show["final_report"])

    with tab2:
        st.markdown("### Output Logs")
        for res in report_to_show["raw_results"]:
            with st.expander(f"Department: {res['source'].upper()}"):
                st.markdown(res["result"])


st.divider()
st.caption(f"App session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

