"""
HypothesisAI - Hypothesis Validation Tool

Internal tool for validating hypotheses using statistical
analysis and research evidence.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from datetime import datetime
import os
import requests
import xml.etree.ElementTree as ET
import re
import openai

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="HypothesisAI",
    page_icon="H",
    layout="wide"
)

# UI stylinggggggggggg
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Main background - Chalk */
    .stApp {
        background-color: #E3D6BF;
    }
    
    .main .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1200px;
    }
    
    /* Status bar - defined later, override early definition */
    
    /* Page header */
    .page-header {
        margin-bottom: 1.5rem;
    }
    
    .page-title {
        font-size: 32px;
        font-weight: 700;
        color: #2B1E1E;
        margin: 0;
    }
    
    .page-subtitle {
        font-size: 15px;
        color: #2B1E1E;
        margin-top: 0.3rem;
        font-weight: 500;
    }
    
    /* Cards - Brook Green background, Dark Chocolate text */
    .card {
        background: #AAB4AE;
        border: 2px solid #9F9679;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #2B1E1E;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #9F9679;
    }
    
    /* Data labels - Dark Chocolate */
    .data-label {
        font-size: 12px;
        font-weight: 700;
        color: #2B1E1E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    .data-value {
        font-size: 15px;
        color: #2B1E1E;
        font-weight: 600;
    }
    
    .data-value-large {
        font-size: 28px;
        color: #2B1E1E;
        font-weight: 700;
    }
    
    /* Metric grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }
    
    .metric-box {
        background: #FFFFFF;
        border: 2px solid #9F9679;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-box.active {
        border-color: #B5728A;
        background: #FFFFFF;
    }
    
    /* Validation result */
    .validation-box {
        background: #FFFFFF;
        border: 2px solid #9F9679;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .validation-box.supported {
        border-left: 5px solid #5a9a6e;
        background: #FFFFFF;
    }
    
    .validation-box.not-supported {
        border-left: 5px solid #B5728A;
        background: #FFFFFF;
    }
    
    .validation-status {
        font-size: 15px;
        font-weight: 700;
        color: #2B1E1E;
        margin-bottom: 0.5rem;
    }
    
    .validation-detail {
        font-size: 14px;
        color: #2B1E1E;
        font-weight: 500;
    }
    
    /* Research table */
    .research-table {
        width: 100%;
        border-collapse: collapse;
        background: #FFFFFF;
    }
    
    .research-table th {
        text-align: left;
        font-size: 12px;
        font-weight: 700;
        color: #2B1E1E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 0.75rem 0.5rem;
        border-bottom: 2px solid #9F9679;
        background: #AAB4AE;
    }
    
    .research-table td {
        padding: 0.75rem 0.5rem;
        border-bottom: 1px solid #9F9679;
        font-size: 14px;
        color: #2B1E1E;
        vertical-align: top;
        background: #FFFFFF;
    }
    
    .research-title {
        font-weight: 600;
        color: #2B1E1E;
    }
    
    .research-abstract {
        font-size: 13px;
        color: #2B1E1E;
        margin-top: 0.3rem;
        line-height: 1.5;
        font-weight: 500;
    }
    
    /* Info items */
    .info-item {
        display: flex;
        flex-direction: column;
    }
    
    /* Buttons - Amaranth bg, WHITE text */
    .stButton > button {
        background: #933B5B !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    .stButton > button:hover {
        background: #B5728A !important;
    }
    
    /* Form elements - White bg, Dark Chocolate text */
    .stTextArea textarea {
        background: #FFFFFF !important;
        border: 2px solid #B5728A !important;
        border-radius: 6px !important;
        font-size: 15px !important;
        color: #2B1E1E !important;
        font-weight: 600 !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #2B1E1E !important;
        opacity: 0.6 !important;
        font-weight: 500 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #933B5B !important;
        box-shadow: 0 0 0 2px rgba(147, 59, 91, 0.2) !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: #FFFFFF !important;
        border: 2px solid #B5728A !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox > div > div > div {
        color: #2B1E1E !important;
        font-weight: 600 !important;
    }
    
    /* Dropdown styling */
    
    /* Main select container */
    .stSelectbox > div > div {
        background: #FFFFFF !important;
        border: 2px solid #B5728A !important;
        border-radius: 6px !important;
    }
    
    /* Selected value text */
    .stSelectbox > div > div > div,
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div {
        color: #2B1E1E !important;
        font-weight: 600 !important;
        background: transparent !important;
    }
    
    /* Dropdown arrow */
    .stSelectbox svg {
        fill: #2B1E1E !important;
    }
    
    /* Dropdown menu container */
    .stSelectbox [data-baseweb="select"] > div {
        background: #FFFFFF !important;
    }
    
    .stSelectbox [data-baseweb="menu"],
    .stSelectbox [data-baseweb="popover"],
    .stSelectbox ul,
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 2px solid #9F9679 !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* All dropdown items - normal state */
    .stSelectbox [role="option"],
    .stSelectbox li,
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] [role="option"],
    [role="listbox"] [role="option"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #2B1E1E !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Dropdown items - hover state */
    .stSelectbox [role="option"]:hover,
    .stSelectbox li:hover,
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [role="option"]:hover,
    [role="listbox"] [role="option"]:hover {
        background: #B5728A !important;
        background-color: #B5728A !important;
        color: #FFFFFF !important;
    }
    
    /* Dropdown items - selected/highlighted state */
    .stSelectbox [aria-selected="true"],
    .stSelectbox [data-highlighted="true"],
    [data-baseweb="menu"] [aria-selected="true"],
    [data-baseweb="menu"] [data-highlighted="true"] {
        background: #E3D6BF !important;
        background-color: #E3D6BF !important;
        color: #2B1E1E !important;
    }
    
    /* Force override any dark backgrounds */
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] * {
        background-color: transparent !important;
    }
    
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div:first-child {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background: #FFFFFF !important;
        color: #2B1E1E !important;
        border: 2px solid #5a9a6e !important;
        border-radius: 6px !important;
    }
    
    .stSuccess p {
        color: #2B1E1E !important;
        font-weight: 600 !important;
    }
    
    .stInfo {
        background: #FFFFFF !important;
        color: #2B1E1E !important;
        border: 2px solid #9F9679 !important;
        border-radius: 6px !important;
    }
    
    .stInfo p {
        color: #2B1E1E !important;
        font-weight: 600 !important;
    }
    
    .stWarning {
        background: #FFFFFF !important;
        color: #2B1E1E !important;
        border: 2px solid #B5728A !important;
        border-radius: 6px !important;
    }
    
    .stWarning p {
        color: #2B1E1E !important;
        font-weight: 600 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        color: #2B1E1E !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #E3D6BF !important;
        border-right: 2px solid #9F9679;
    }
    
    /* Code blocks */
    .stCode, pre, .stCodeBlock {
        background: #FFFFFF !important;
        border: 1px solid #9F9679 !important;
        border-radius: 4px !important;
        padding: 0.5rem !important;
    }
    
    code {
        color: #2B1E1E !important;
        background: transparent !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        line-height: 1.8 !important;
    }
    
    /* Remove all pill styling */
    .stMultiSelect [data-baseweb="tag"] {
        background: #B5728A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 2px solid #9F9679 !important;
        border-radius: 6px !important;
    }
    
    /* All text elements - Dark Chocolate for high contrast */
    p, span, div, label {
        color: #2B1E1E;
    }
    
    /* Source card */
    .source-card {
        background: #FFFFFF;
        border: 2px solid #9F9679;
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Card headers - Brook Green bg, Dark Chocolate text */
    .card-header {
        font-size: 14px;
        font-weight: 700;
        color: #2B1E1E !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
        padding: 0.5rem 0.75rem;
        background: #AAB4AE;
        border-radius: 4px;
        border-bottom: none;
    }
    
    /* Variable list - Brook Green bg, Dark Chocolate text */
    .var-list {
        background: #FFFFFF;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #9F9679;
    }
    
    .var-item {
        color: #2B1E1E;
        font-weight: 600;
        font-size: 13px;
        padding: 0.4rem 0.6rem;
        margin: 2px 0;
        border-radius: 4px;
        cursor: default;
    }
    
    .var-item:hover {
        background: #B5728A;
        color: #FFFFFF;
    }
    
    /* Status bar - Thulian Pink bg, WHITE text */
    .status-bar {
        background: #B5728A !important;
        color: #FFFFFF !important;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #FFFFFF !important;
    }
    
    .status-item span {
        color: #FFFFFF !important;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #FFFFFF;
        border-radius: 50%;
    }
    
    /* Input fields - White bg, Dark Chocolate text */
    input, textarea, select {
        background: #FFFFFF !important;
        color: #2B1E1E !important;
        font-weight: 600 !important;
    }
    
    /* Additional dropdown overrides to ensure white backgrounds */
    [data-baseweb="menu"] li,
    [data-baseweb="select"] li,
    div[role="listbox"] li {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #2B1E1E !important;
        font-weight: 500 !important;
    }
    
    [data-baseweb="menu"] li:hover,
    [data-baseweb="select"] li:hover,
    div[role="listbox"] li:hover {
        background: #B5728A !important;
        background-color: #B5728A !important;
        color: #FFFFFF !important;
    }
    
    /* Override any dark theme from Streamlit */
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] > div > div,
    [data-baseweb="popover"] ul {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }
    
    /* Ensure visible labels on all form elements */
    .stTextArea label, .stSelectbox label, .stTextInput label {
        color: #2B1E1E !important;
        font-weight: 700 !important;
        font-size: 13px !important;
    }
    
    /* Final override for any remaining dark elements */
    [data-theme="dark"] [data-baseweb="menu"],
    [data-theme="dark"] [data-baseweb="popover"],
    [data-theme="dark"] [role="listbox"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================
DATASETS = {
    "Student Performance (Math)": {
        "path": "hypothesisdataset/student-mat.csv",
        "separator": ";",
        "description": "Math course achievement data"
    },
    "Student Performance (Portuguese)": {
        "path": "hypothesisdataset/student-por.csv",
        "separator": ";",
        "description": "Portuguese course achievement data"
    }
}

# ============================================================
# VARIABLE DESCRIPTIONS
# ============================================================
VARIABLE_DESCRIPTIONS = {
    "age": "Student's age in years",
    "Medu": "Mother's education level (0=none, 4=higher)",
    "Fedu": "Father's education level (0=none, 4=higher)",
    "traveltime": "Time taken to travel to school",
    "studytime": "Weekly study time",
    "failures": "Number of past class failures",
    "famrel": "Quality of family relationships",
    "freetime": "Free time after school",
    "goout": "How often student goes out with friends",
    "Dalc": "Workday alcohol consumption",
    "Walc": "Weekend alcohol consumption",
    "health": "Current health status",
    "absences": "Number of school absences",
    "G1": "First period grade",
    "G2": "Second period grade",
    "G3": "Final grade"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(dataset_name):
    """Load selected dataset."""
    try:
        config = DATASETS[dataset_name]
        df = pd.read_csv(config["path"], sep=config["separator"])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df, numeric_cols, config["description"]
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, [], ""


def infer_expected_direction(hypothesis):
    """Infer expected relationship direction from hypothesis text."""
    hypothesis_lower = hypothesis.lower()
    
    positive_keywords = [
        'more', 'increase', 'higher', 'greater', 'improve', 'better',
        'raise', 'enhance', 'boost', 'grow', 'leads to', 'results in',
        'positively', 'correlates with'
    ]
    
    negative_keywords = [
        'less', 'decrease', 'lower', 'reduce', 'fewer', 'worse',
        'decline', 'diminish', 'drop', 'fall', 'negatively'
    ]
    
    positive_count = sum(1 for kw in positive_keywords if kw in hypothesis_lower)
    negative_count = sum(1 for kw in negative_keywords if kw in hypothesis_lower)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    return 'unknown'


def get_observed_direction(correlation):
    """Determine observed relationship direction."""
    if correlation > 0.05:
        return 'positive'
    elif correlation < -0.05:
        return 'negative'
    return 'neutral'


def validate_hypothesis(expected, observed):
    """Compare expected vs observed direction."""
    if expected == 'unknown':
        return 'undetermined', "Could not infer expected direction"
    
    if expected == observed:
        return 'supported', f"Data confirms {expected} relationship"
    elif observed == 'neutral':
        return 'not_supported', f"Expected {expected}, observed no significant correlation"
    else:
        return 'not_supported', f"Expected {expected}, observed {observed}"


def calculate_statistics(df, independent_var, dependent_var):
    """Calculate correlation, regression, and p-value."""
    x = df[independent_var].values
    y = df[dependent_var].values
    
    correlation, p_value = stats.pearsonr(x, y)
    
    X = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    r_squared = model.score(X, y)
    
    return {
        'correlation': round(correlation, 4),
        'p_value': round(p_value, 4),
        'r_squared': round(r_squared, 4),
        'slope': round(model.coef_[0], 4),
        'intercept': round(model.intercept_, 4),
        'predictions': predictions
    }


def get_confidence_level(correlation, p_value):
    """Determine confidence level based on correlation and p-value."""
    abs_corr = abs(correlation)
    if p_value < 0.01 and abs_corr >= 0.5:
        return "High"
    elif p_value < 0.05 and abs_corr >= 0.3:
        return "Medium"
    return "Low"


def process_hypothesis_for_search(hypothesis):
    """Convert user hypothesis into academic search keywords."""
    # Common mappings
    mappings = {
        "grades": "academic performance",
        "marks": "academic achievement",
        "score": "test scores",
        "study": "study time",
        "alcohol": "alcohol consumption",
        "drink": "alcohol consumption",
        "absent": "school absenteeism",
        "friends": "peer influence",
        "family": "family environment",
        "health": "student health"
    }
    
    # Process text
    clean_text = hypothesis.lower()
    for word, replacement in mappings.items():
        if word in clean_text:
            clean_text = clean_text.replace(word, replacement)
            
    # Remove filler words (basic list)
    stop_words = [
        'students', 'with', 'who', 'have', 'higher', 'lower', 'better', 'worse',
        'less', 'more', 'achieve', 'get', 'are', 'is', 'the', 'a', 'an', 'in',
        'to', 'for', 'of', 'and', 'but', 'or', 'if', 'then', 'than', 'relationship',
        'between', 'affect', 'impact', 'influence'
    ]
    
    words = re.findall(r'\b\w+\b', clean_text)
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Ensure academic context if missing
    if "student" not in keywords and "academic" not in keywords and "education" not in keywords:
        keywords.append("education")
        
    return ' '.join(list(set(keywords))[:5])  # Limit to unique top 5 keywords


def get_ai_explanation(hypothesis, stats_result, confidence, validation_result):
    """Get explanation from OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    
    if not api_key:
        return "OpenAI API Key not found. Please set OPENAI_API_KEY in your environment variables."
    
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    Act as a statistical research assistant. Explain these results to a user.
    
    Context:
    Hypothesis: "{hypothesis}"
    Results:
    - Correlation: {stats_result['correlation']} ({'Positive' if stats_result['correlation'] > 0 else 'Negative'})
    - P-Value: {stats_result['p_value']}
    - R-Squared: {stats_result['r_squared']}
    - Confidence: {confidence}
    - Verdict: {validation_result.upper()}
    
    Instructions:
    1. Explain the result in simple human terms.
    2. Provide a practical interpretation (what does this mean for students/teachers?).
    3. Suggest one relevant follow-up variable to test from the dataset context (e.g. absences, failures, studytime).
    4. Keep it under 100 words.
    5. Be objective. Math is the source of truth.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Explanation unavailable: {str(e)}"


def fetch_arxiv_papers(hypothesis):
    """Fetch related research papers from arXiv API."""
    # Preprocess query
    query_terms = process_hypothesis_for_search(hypothesis)
    query = '+'.join(query_terms.split())
    
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            summary = entry.find('atom:summary', ns)
            published = entry.find('atom:published', ns)
            
            if title is not None:
                year = published.text[:4] if published is not None else "N/A"
                abstract_text = summary.text.strip() if summary is not None else ""
                sentences = abstract_text.split('.')
                short_abstract = '. '.join(sentences[:2]) + '.' if sentences else ""
                
                papers.append({
                    'title': title.text.strip().replace('\n', ' '),
                    'year': year,
                    'abstract': short_abstract[:300]
                })
        
        return papers
    except:
        return []


def save_to_history(hypothesis, dataset, independent_var, dependent_var, 
                    correlation, p_value, confidence, validation_result):
    """Save result to history.csv."""
    history_file = "history.csv"
    new_entry = pd.DataFrame({
        'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M")],
        'hypothesis': [hypothesis[:50] + '...' if len(hypothesis) > 50 else hypothesis],
        'dataset': [dataset.split('(')[1].replace(')', '') if '(' in dataset else dataset],
        'x_var': [independent_var],
        'y_var': [dependent_var],
        'correlation': [correlation],
        'p_value': [p_value],
        'confidence': [confidence],
        'result': [validation_result]
    })
    
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        history = pd.concat([history, new_entry], ignore_index=True)
    else:
        history = new_entry
    
    history.to_csv(history_file, index=False)
    return history


def load_history():
    """Load hypothesis history."""
    if os.path.exists("history.csv"):
        return pd.read_csv("history.csv")
    return pd.DataFrame()


def create_regression_plot(df, independent_var, dependent_var, predictions):
    """Create scatter plot with regression line using new palette."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    
    ax.scatter(df[independent_var], df[dependent_var], 
               alpha=0.7, color='#AAB4AE', s=35, edgecolors='#9F9679', linewidth=0.5)
    
    
    sorted_idx = df[independent_var].argsort()
    ax.plot(df[independent_var].iloc[sorted_idx], predictions[sorted_idx], 
            color='#933B5B', linewidth=2.5)
    
    ax.set_xlabel(independent_var, fontsize=12, color='#933B5B', fontweight='600')
    ax.set_ylabel(dependent_var, fontsize=12, color='#933B5B', fontweight='600')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9F9679')
    ax.spines['bottom'].set_color('#9F9679')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(colors='#933B5B', labelsize=10)
    
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    plt.tight_layout()
    return fig


def create_trend_chart(history):
    """Create correlation trend bar chart using new palette."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    
    x = range(1, len(history) + 1)
    # Brook Green for supported, Thulian Pink for not supported
    colors = ['#AAB4AE' if r == 'supported' else '#B5728A' 
              for r in history.get('result', ['supported'] * len(history))]
    
    ax.bar(x, history['correlation'], color=colors, edgecolor='#9F9679', linewidth=1.5)
    ax.axhline(y=0, color='#9F9679', linewidth=2)
    
    ax.set_xlabel('Test #', fontsize=11, color='#933B5B', fontweight='600')
    ax.set_ylabel('r', fontsize=11, color='#933B5B', fontweight='600')
    ax.set_xticks(x)
    ax.set_ylim(-1, 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9F9679')
    ax.spines['bottom'].set_color('#9F9679')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(colors='#933B5B', labelsize=10)
    
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    plt.tight_layout()
    return fig


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Page header
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">HypothesisAI</h1>
        <p class="page-subtitle">Validate hypotheses using real datasets and research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset selection
    selected_dataset = st.selectbox(
        "Select Dataset",
        options=list(DATASETS.keys())
    )
    
    df, numeric_cols, dataset_desc = load_dataset(selected_dataset)
    
    if df is None:
        return
    
    # Status bar
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-item">
            <span class="status-dot"></span>
            <span>Dataset loaded</span>
        </div>
        <div class="status-item">
            <span>{len(df):,} rows</span>
        </div>
        <div class="status-item">
            <span>{len(numeric_cols)} numeric variables</span>
        </div>
        <div class="status-item">
            <span>Last updated today</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout columns
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        # Hypothesis input card
        st.markdown("""
        <div class="card">
            <div class="card-header">Hypothesis Input</div>
        """, unsafe_allow_html=True)
        
        hypothesis = st.text_area(
            "Enter your hypothesis",
            placeholder="Example: Students with higher study time achieve better grades",
            height=80
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Variables card
        st.markdown("""
        <div class="card">
            <div class="card-header">Variable Selection</div>
        """, unsafe_allow_html=True)
        
        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<div class="data-label">Independent Variable (X)</div>', unsafe_allow_html=True)
            independent_var = st.selectbox(
                "Select X variable",
                options=numeric_cols,
                index=numeric_cols.index('studytime') if 'studytime' in numeric_cols else 0,
                format_func=lambda x: f"{x} – {VARIABLE_DESCRIPTIONS.get(x, '')}"
            )
        with v2:
            st.markdown('<div class="data-label">Dependent Variable (Y)</div>', unsafe_allow_html=True)
            dependent_var = st.selectbox(
                "Select Y variable",
                options=numeric_cols,
                index=numeric_cols.index('G3') if 'G3' in numeric_cols else 0,
                format_func=lambda x: f"{x} – {VARIABLE_DESCRIPTIONS.get(x, '')}"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Run button
        run_analysis = st.button("Run Experiment", use_container_width=True)
    
    with col_side:
        # Data source card
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Data Source</div>
            <div class="source-card">
                <div class="info-item" style="margin-bottom: 0.75rem;">
                    <div class="data-label">Dataset</div>
                    <div class="data-value">{selected_dataset.split('(')[1].replace(')', '') if '(' in selected_dataset else selected_dataset}</div>
                </div>
                <div class="info-item" style="margin-bottom: 0.75rem;">
                    <div class="data-label">Total Records</div>
                    <div class="data-value">{len(df):,}</div>
                </div>
                <div class="info-item">
                    <div class="data-label">Description</div>
                    <div class="data-value">{dataset_desc}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Variables list - plain text, no pills
        # Variables list - plain text with descriptions
        var_items = ''.join([
            f'<div class="var-item"><strong>{col}</strong><br><span style="font-weight:400; font-size:11px;">{VARIABLE_DESCRIPTIONS.get(col, "")}</span></div>' 
            for col in numeric_cols
        ])
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Available Variables</div>
            <div class="var-list">
                {var_items}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Results
    if run_analysis:
        if not hypothesis.strip():
            st.warning("Please enter a hypothesis to continue.")
        else:
            st.markdown('<div class="section-header">Experiment Results</div>', unsafe_allow_html=True)
            
            # Calculate statistics
            stats_result = calculate_statistics(df, independent_var, dependent_var)
            confidence = get_confidence_level(stats_result['correlation'], stats_result['p_value'])
            
            # Validation
            expected_dir = infer_expected_direction(hypothesis)
            observed_dir = get_observed_direction(stats_result['correlation'])
            validation_result, validation_msg = validate_hypothesis(expected_dir, observed_dir)
            
            # Validation box
            validation_class = "supported" if validation_result == "supported" else "not-supported"
            status_text = "HYPOTHESIS SUPPORTED" if validation_result == "supported" else "HYPOTHESIS NOT SUPPORTED"
            
            st.markdown(f"""
            <div class="validation-box {validation_class}">
                <div class="validation-status">{status_text}</div>
                <div class="validation-detail">
                    Expected: {expected_dir.upper()} | Observed: {observed_dir.upper()} | {validation_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="data-label">Correlation (r)</div>
                    <div class="data-value-large">{stats_result['correlation']}</div>
                </div>
                <div class="metric-box">
                    <div class="data-label">P-Value</div>
                    <div class="data-value-large">{stats_result['p_value']}</div>
                </div>
                <div class="metric-box">
                    <div class="data-label">R-Squared</div>
                    <div class="data-value-large">{stats_result['r_squared']}</div>
                </div>
                <div class="metric-box {'active' if confidence == 'High' else ''}">
                    <div class="data-label">Confidence</div>
                    <div class="data-value-large">{confidence}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # Regression plot
            col_plot, col_stats = st.columns([2, 1])
            
            with col_plot:
                fig = create_regression_plot(df, independent_var, dependent_var, stats_result['predictions'])
                st.pyplot(fig)
                plt.close()
            
            with col_stats:
                st.markdown(f"""
                <div class="card">
                    <div class="card-header">Regression Details</div>
                    <div class="source-card">
                        <div class="info-item" style="margin-bottom: 0.5rem;">
                            <div class="data-label">Slope</div>
                            <div class="data-value">{stats_result['slope']}</div>
                        </div>
                        <div class="info-item" style="margin-bottom: 0.5rem;">
                            <div class="data-label">Intercept</div>
                            <div class="data-value">{stats_result['intercept']}</div>
                        </div>
                        <div class="info-item">
                            <div class="data-label">Sample Size</div>
                            <div class="data-value">{len(df)}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # AI Explanation Section
            st.markdown('<div class="section-header">AI Explanation</div>', unsafe_allow_html=True)
            
            with st.spinner("Generating explanation..."):
                explanation = get_ai_explanation(
                    hypothesis, stats_result, confidence, validation_result
                )
            
            st.markdown(f"""
            <div class="card">
                <div class="card-header" style="color: #2B1E1E !important;">Interpretation</div>
                <div style="color: #2B1E1E; line-height: 1.6; font-size: 15px;">
                    {explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Research Evidence
            st.markdown('<div class="section-header">Research Evidence</div>', unsafe_allow_html=True)
            
            with st.spinner("Querying arXiv database..."):
                papers = fetch_arxiv_papers(hypothesis)
            
            if papers:
                st.markdown("""
                <div class="card">
                    <table class="research-table">
                        <tr>
                            <th style="width: 70px;">Year</th>
                            <th>Title & Abstract</th>
                        </tr>
                """, unsafe_allow_html=True)
                
                for paper in papers:
                    st.markdown(f"""
                        <tr>
                            <td><strong>{paper['year']}</strong></td>
                            <td>
                                <div class="research-title">{paper['title']}</div>
                                <div class="research-abstract">{paper['abstract']}</div>
                            </td>
                        </tr>
                    """, unsafe_allow_html=True)
                
                st.markdown("</table></div>", unsafe_allow_html=True)
            else:
                st.info("No related papers found in arXiv database.")
            
            # Save to history
            save_to_history(
                hypothesis, selected_dataset, independent_var, dependent_var,
                stats_result['correlation'], stats_result['p_value'], confidence, validation_result
            )
            st.success("Experiment saved to history")
    
    # History section
    history = load_history()
    
    if len(history) > 0:
        st.markdown('<div class="section-header">Experiment History</div>', unsafe_allow_html=True)
        
        st.dataframe(history, use_container_width=True, hide_index=True)
        
        if len(history) >= 2:
            trend_fig = create_trend_chart(history)
            st.pyplot(trend_fig)
            plt.close()


if __name__ == "__main__":
    main()
