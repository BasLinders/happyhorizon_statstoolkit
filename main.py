import streamlit as st
from importlib import import_module

st.set_page_config(
    page_title="Happy Horizon Experimentation Toolkit",
    page_icon="📈",
)

st.title("Happy Horizon Experimentation Toolkit")

# Create two columns
col1, col2 = st.columns([1, 2])

# Dictionary to map app names to their respective modules
apps = {
    "Bayesian calculator": "pages.bayesian_calculator.py",
    "Frequentist calculator": "pages.frequentist_calculator.py",
    "Continuous data calculator": "pages.continuous_calculator.py",
    "Interaction Effect calculator": "pages.interaction_calculator.py",
    "SRM calculator": "pages.srm_calculator.py",
    "Sample Size / MDE calculator": "pages.sample_size_calculator.py"
}

# Left column: Dropdown menu for selecting an app
with col1:
    selected_app = st.selectbox("Choose an app", list(apps.keys()))
    st.sidebar.success("Select an app above.")

# Right column: Introduction text
with col2:
    st.write("""
    ## Experimentation Toolkit
    This is the main page for the Happy Horizon Experimentation Toolkit. You can navigate to individual apps using the dropdown on the left.

    ### Features
    - Bayesian calculator: Calculate the probability of a winner and make a business case
    - Frequentist calculator: Calculate significance for your test variant as tangible proof for a winner
    - Continuous data calculator: Analyze metrics such as AOV, items per transaction for your experiment
    - Interaction Effect calculator: Verify if your experiments negatively impacted eachother or not
    - SRM calculator: Identify if your visitors were distributed as expected in your experiment
    - Sample Size / MDE calculator: See how long your experiment has to run to reach an effect before you can draw conclusions

    ### How to Use
    - Select a page from the dropdown to view different tools.
    - Each page contains a single tool for the purposes described above.

    ### About
    This toolkit has been created for the purposes of analyzing data from online controlled experiments ('A/B tests').

    ### Contact
    For more information, visit [HappyHorizon.com](https://happyhorizon.com).
    """)

st.write("")
linkedin_url = "https://www.linkedin.com/in/blinders/"
happyhorizon_url = "https://happyhorizon.com/"
footnote_text = f"""Engineered and developed by <a href="{linkedin_url}" target="_blank">Bas Linders</a> @<a href="{happyhorizon_url}" target="_blank">Happy Horizon.</a>"""
st.markdown(footnote_text, unsafe_allow_html=True)
st.write("")

# Load the selected module
app = import_module(apps[selected_app])
app.run()