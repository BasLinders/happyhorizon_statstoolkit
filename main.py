import streamlit as st

st.set_page_config(
    page_title="Happy Horizon Experimentation Toolkit",
    page_icon="📈",
)

st.title("Happy Horizon Experimentation Toolkit")

st.write("""
## Experimentation Toolkit
This is the main page for the Happy Horizon Experimentation Toolkit. You can navigate to individual apps using the sidebar.

### Features
- **Bayesian calculator**: Calculate the probability of a winner and make a business case
- **Frequentist calculator**: Calculate significance for your test variant as tangible proof for a winner
- **Continuous data calculator**: Analyze metrics such as AOV, items per transaction for your experiment
- **Interaction Effect calculator**: Verify if your experiments negatively impacted each other or not
- **SRM calculator**: Identify if your visitors were distributed as expected in your experiment
- **Sample Size / MDE calculator**: See how long your experiment has to run to reach an effect before you can draw conclusions

### How to Use
- Select a page from the sidebar to view different tools.
- Each page contains a single tool for the purposes described above.

### About
This toolkit has been created for the purposes of analyzing data from online controlled experiments ('A/B tests').

### Contact
For more information, visit [HappyHorizon.com](https://happyhorizon.com).
""")

linkedin_url = "https://www.linkedin.com/in/blinders/"
happyhorizon_url = "https://happyhorizon.com/"
footnote_text = f"""Engineered and developed by <a href="{linkedin_url}" target="_blank">Bas Linders</a> @<a href="{happyhorizon_url}" target="_blank">Happy Horizon.</a>"""
st.markdown(footnote_text, unsafe_allow_html=True)