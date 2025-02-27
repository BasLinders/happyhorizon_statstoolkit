import streamlit as st
import importlib.util
import os

st.set_page_config(
    page_title="Happy Horizon Experimentation Toolkit",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Define hidden pages (not included in sidebar)
hidden_pages = {
    "mab_test": "MAB Test"
}

# Ensure session state exists
if "current_page" not in st.session_state:
    st.session_state.current_page = None
if "page_loaded" not in st.session_state:
    st.session_state.page_loaded = False

# Get the query parameter from the URL
query_params = st.query_params
page = query_params.get("page", [None])[0]

# Function to execute hidden page code
def execute_hidden_page(page_name):
    page_path = f"hidden_pages/{page_name}.py"
    if os.path.exists(page_path):
        with open(page_path, "r") as f:
            code = f.read()
            exec(code)
    else:
        st.error("Page not found.")

# Update session state and query parameters
if page in hidden_pages and not st.session_state.page_loaded:
    st.session_state.current_page = page
    st.query_params.set_all(page=page)  # Correct way to set query parameters
    st.session_state.page_loaded = True
elif st.session_state.current_page is None:
    st.query_params.clear()  # Correct way to clear query parameters

# Conditional Rendering
if st.session_state.current_page:
    st.title(hidden_pages[st.session_state.current_page])
    execute_hidden_page(st.session_state.current_page)
else:
    # Main Page UI
    logo_url = "https://cdn.homerun.co/49305/hh-woordmerk-rgb-blue-met-discriptor1666785216logo.png"
    st.image(logo_url, width=200)

    st.title("Happy Horizon Experimentation Toolkit")
    st.write("### <span style='color: orange;'>v1.5.5 (beta)</span>", unsafe_allow_html=True)
    st.write("""
    This is the main page for the Happy Horizon Experimentation Toolkit. You can navigate to individual apps using the sidebar.

    ### What you're looking at
    This toolkit has been created for the purposes of analyzing data from online controlled experiments ('A/B tests') to learn from and better understand user behavior.  

    ### Features
    <span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Bayesian calculator**: Calculate the probability of a winner and make a business case<br>
    <span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Frequentist calculator**: Calculate significance for your test variant as tangible proof for a winner<br>
    <span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Continuous data calculator**: Analyze metrics such as revenue / items per transaction<br>
    <span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Experimentation growth calculator**: Calculate annual compound growth potential for experimentation<br>
    <span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Interaction Effect calculator**: Verify if your experiments negatively impacted each other or not<br>
    <span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**SRM calculator**: Identify if your visitors were distributed as expected in your experiment<br>
    <span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Sample Size / MDE calculator**: Calculate the runtime to reach an effect

    ### How to Use
    - Select a page from the sidebar to view different tools.
    - Each page contains a single tool for the purposes described above.

    ### About
    Happy Horizon is a creative digital agency of experts in strategic thinking, analysis, creativity, digital services and technology.
    """, unsafe_allow_html=True)

    linkedin_url = "https://www.linkedin.com/in/blinders/"
    happyhorizon_url = "https://happyhorizon.com/"
    footnote_text = f"""Engineered and developed by <a href="{linkedin_url}" target="_blank">Bas Linders</a> @<a href="{happyhorizon_url}" target="_blank">Happy Horizon.</a>"""
    st.markdown(footnote_text, unsafe_allow_html=True)
