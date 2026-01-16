import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import io
import re
import numpy as np
from fpdf import FPDF
import tempfile
import textwrap
from datetime import timedelta
import requests

# --- SKLEARN IMPORTS FOR FORECASTING ---
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- 1. GLOBAL UI & CONFIG ---
st.set_page_config(page_title="Strategic BI Analyst Pro", layout="wide", page_icon="💎")

# Fix for Matplotlib GUI crash on some servers
plt.switch_backend('Agg') 

def clean_text_for_pdf(text):
    """Sanitizes text to prevent FPDF latin-1 errors."""
    if not isinstance(text, str):
        return str(text)
    # Replace specific characters that commonly break FPDF
    text = text.replace('\u2013', '-').replace('\u2014', '--') # Replace dashes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Replace smart quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Replace smart quotes
    text = text.replace('•', '-') # Replace bullets
    
    # Force convert to latin-1, replacing unknown characters with '?'
    return text.encode('latin-1', 'replace').decode('latin-1')

def apply_plt_styles():
    """Forces professional chart styling."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.autolayout': True,
        'axes.titlepad': 20,
        'xtick.major.pad': 8
    })

if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None

def clean_and_space_text(text, width=90):
    """Deep-cleans AI text to fix word mashing."""
    if not text: return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([.!?])\s*([•*]|\d\.)', r'\1\n\n\2', text)
    text = re.sub(r'^(###|\d\.|\*|-|Insight:|Summary:)\s*', '• ', text, flags=re.MULTILINE)
    
    lines = [textwrap.fill(line, width=width) for line in text.split('\n')]
    return "\n".join(lines).strip()

def sanitize_logic(code):
    """Corrects AI typos and strips Markdown."""
    code = re.sub(r"```python\n?", "", code)
    code = code.replace("```", "").strip()
    code = code.replace("to_to_datetime", "to_datetime")
    code = code.replace("pd.pd.", "pd.")
    return code

def get_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf.getvalue()

# --- 2. MULTI-LLM WRAPPER ---
def call_llm(provider, api_key, model, prompt):
    try:
        if provider == "Gemini":
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel(model)
            response = m.generate_content(prompt)
            return response.text
        
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        endpoints = {
            "Groq": "https://api.groq.com/openai/v1/chat/completions",
            "DeepSeek": "https://api.deepseek.com/v1/chat/completions",
            "OpenRouter (Qwen)": "https://openrouter.ai/api/v1/chat/completions",
            "Cerebras": "https://api.cerebras.ai/v1/chat/completions"
        }
        url = endpoints[provider]
        data = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
        res = requests.post(url, headers=headers, json=data)
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. SIDEBAR: CONFIG & PDF EXPORT ---
with st.sidebar:
    st.title("⚙️ AI Analytics Engine")
    provider = st.radio("Select Provider", ["Gemini", "Groq", "DeepSeek", "OpenRouter (Qwen)", "Cerebras"])
    api_key = st.text_input(f"🔑 {provider} API Key", type="password")
    
    models = {
        "Gemini": ["gemini-3-flash-preview","gemini-2.0-flash-exp", "gemini-1.5-flash"],
        "Groq": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
        "OpenRouter (Qwen)": ["qwen/qwen-2.5-coder-32b-instruct", "qwen/qwen-2-72b-instruct"],
        "Cerebras": ["llama3.1-70b", "llama3.1-8b"]
    }
    model_choice = st.selectbox("Model", models[provider])

    uploaded_file = st.file_uploader("📂 Upload Business Data", type=['csv', 'xlsx'])
    
    st.divider()
    if st.button("📄 Export PDF Report"):
        if not st.session_state.messages:
            st.error("No results to export.")
        else:
            try:
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", 'B', 18)
                pdf.cell(200, 15, txt="Business Insight Strategic Report", ln=True, align='C')
                
                for m in st.session_state.messages:
                    if m["role"] == "assistant":
                        pdf.ln(10); pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, txt="Strategic Insight:", ln=True)
                        pdf.set_font("Arial", size=10)
                        
                        # --- CLEAN TEXT ---
                        raw_text = m.get('insight', '')
                        clean_pdf_text = clean_text_for_pdf(raw_text) 
                        pdf.multi_cell(0, 7, txt=clean_pdf_text)
                        
                        if "image" in m:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                tmp.write(m["image"])
                                pdf.image(tmp.name, x=15, w=170)
                                
                # --- FIXED: Explicit encoding to 'latin-1' to return bytes ---
                pdf_data = pdf.output(dest='S').encode('latin-1')
                
                st.download_button(
                    label="📥 Download PDF", 
                    data=pdf_data, 
                    file_name="BI_Strategic_Report.pdf", 
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"PDF Error: {e}")

    if st.button("🗑️ Reset Workspace"):
        st.session_state.clear(); st.rerun()

# --- 4. DATA ENGINE ---
if api_key and uploaded_file:
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.title()
        for col in df.columns:
            if any(x in col.lower() for x in ['date', 'time', 'period']):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        st.subheader("📊 Dataset Explorer")
        st.dataframe(df.head(5), use_container_width=True)

        st.write("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🪄 Strategic Questions"):
                prompt = f"Using columns {df.columns.tolist()}, generate 3 strategic business questions. One per line."
                res_text = call_llm(provider, api_key, model_choice, prompt)
                st.session_state.suggestions = [l.strip() for l in res_text.split('\n') if '?' in l][:3]
                st.rerun()
        with c2:
            if st.button("📝 Deep Insights"):
                prompt = f"Analyze {df.columns.tolist()}. Provide exactly 5 deep business bullet points on separate lines. Fix font mashing."
                res_text = call_llm(provider, api_key, model_choice, prompt)
                st.session_state.data_summary = clean_and_space_text(res_text)
                st.rerun()
        with c3:
            if st.button("🔮 90-Day Trend Predictor"):
                st.session_state.pending_query = "Create a continuous 90-day linear forecast using Sklearn. Show numeric labels."

        if st.session_state.data_summary:
            st.info(st.session_state.data_summary)

        if st.session_state.suggestions:
            for i, sug in enumerate(st.session_state.suggestions):
                if st.button(f"🔍 {sug}", key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_query = sug

        st.write("---")

        # --- 5. EXECUTION & VISUALIZATION ---
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if "image" in m: st.image(m["image"])
                if "insight" in m: st.success(f"💡 {m['insight']}")

        active_query = st.session_state.get("pending_query") or st.chat_input("Ask about sales, trends, or segments...")
        if "pending_query" in st.session_state: del st.session_state["pending_query"]

        if active_query:
            st.session_state.messages.append({"role": "user", "content": active_query})
            with st.chat_message("user"): st.markdown(active_query)

            with st.chat_message("assistant"):
                try:
                    prompt = (f"Columns: {df.columns.tolist()}. Task: {active_query}.\n"
                              f"STRICT RULES:\n"
                              f"1. DATA LABELS: You MUST show numeric values on every visual (use ax.bar_label or plt.text).\n"
                              f"2. NUMERIC GUARD: Use numeric_only=True and drop NaNs before math.\n"
                              f"3. VISUALS: If categories > 8, use barh (horizontal). Rotate labels 45 deg.\n"
                              f"4. OUTPUT: Python code block first, then '###', then 5 professional bullet points.")
                    
                    raw_out = call_llm(provider, api_key, model_choice, prompt)
                    parts = raw_out.split("###") if "###" in raw_out else [raw_out, "Analysis complete."]
                    code = sanitize_logic(parts[0])
                    
                    apply_plt_styles()
                    plt.close('all') # Clear previous plots
                    
                    # SYSTEM REINFORCEMENT
                    exec_env = {
                        'df': df, 'pd': pd, 'plt': plt, 'sns': sns, 'np': np, 
                        'timedelta': timedelta, 'LinearRegression': LinearRegression, 
                        'train_test_split': train_test_split, 'io': io, 're': re
                    }
                    exec(code, {"__builtins__": __builtins__}, exec_env)
                    
                    fig = plt.gcf()
                    clean_insight = clean_and_space_text(parts[1])
                    new_msg = {"role": "assistant", "content": "Analysis Result", "insight": clean_insight}
                    
                    if fig.get_axes():
                        st.pyplot(fig)
                        new_msg["image"] = get_image_bytes(fig)
                    
                    st.session_state.messages.append(new_msg)
                    st.rerun()
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    with st.expander("Show System Logic"): st.code(code)

    except Exception as e:
        st.error(f"Critical System Error: {e}")
else:
    st.info("👋 Select an AI provider, enter your key, and upload data to begin.")