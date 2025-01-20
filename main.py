import streamlit as st

st.set_page_config(
    page_title="Happy Horizon Experimentation Toolkit",
    page_icon="📈",
)

logo_url = "https://cdn.homerun.co/49305/hh-woordmerk-rgb-blue-met-discriptor1666785216logo.png"
st.image(logo_url, width=200) # alternative: use_column_width=True

st.title("Happy Horizon Experimentation Toolkit")
st.write("### <span style='color: orange;'>v1.5.0 (beta)</span>", unsafe_allow_html=True)
st.write("""
This is the main page for the Happy Horizon Experimentation Toolkit. You can navigate to individual apps using the sidebar.

### What you're looking at
This toolkit has been created for the purposes of analyzing data from online controlled experiments ('A/B tests') to learn from and better understand user behavior.  

### Features
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Bayesian calculator**: Calculate the probability of a winner and make a business case<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Frequentist calculator**: Calculate significance for your test variant as tangible proof for a winner<br>
<span style='color:#009900; font-weight: 600; margin-right: 6px;'>&#10003;</span>**Continuous data calculator**: Analyze metrics such as revenue / items per transaction<br>
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