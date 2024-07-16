import streamlit as st
from scipy.stats import norm
import pandas as pd
import numpy as np

def run():
    st.set_page_config(
        page_title="Sample Size / MDE calculator",
        page_icon="🔢",
    )

    st.title("Sample Size Calculator")
    """
    This calculator provides you with an representative sample size and Minimum Detectable Effect for your online experiment. Enter the values below to start.

    Happy learning!
    """
    col1, col2 = st.columns(2)
    # Inputs
    with col1:
        baseline_visitors = st.number_input("Amount of visitors per week:", min_value=0, step=1)
        risk = st.number_input("In %, enter the desired confidence rate (e.g. 95)?", min_value=0.0, max_value=100.0, step=0.1)
    with col2:
        baseline_conversions = st.number_input("Number of conversions per week:", min_value=0, step=1)
        trust = st.number_input("In %, enter the minimum trustworthiness (e.g. 80)", min_value=0.0, max_value=100.0, step=0.1)

    tails = st.selectbox("Do you want to know if B is better than A, or also the other way around (i.e. verify a negative effect)?", ('B better than A', 'A better than B'))

    st.write("")
    if st.button("Calculate Sample size and MDE"):
        if any([baseline_visitors <= 0, baseline_conversions <= 0, risk <= 0, trust <= 0, tails not in ['B better than A', 'A better than B']]):
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)
        else:
            alpha = 1 - (risk / 100)
            power = trust / 100

            # Calculate baseline conversion rate
            baseline_rate = baseline_conversions / baseline_visitors

            # Z-scores for confidence and power
            if tails == 'A better than B':
                z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed
            else:
                z_alpha = norm.ppf(1 - alpha)
            z_power = norm.ppf(power)

            # Weekly increments
            weeks = range(1, 7)  # For 6 weeks
            weekly_visitors_increase = np.ceil(baseline_visitors / 2)

            # Prepare a list to store the results for each week
            results = []
            for week in weeks:
                visitors_per_variant = int(weekly_visitors_increase * week)
                variant_cr = baseline_rate  # Assuming constant conversion rate over weeks for simplicity
                
                # Solving for MDE
                se = np.sqrt(2 * variant_cr * (1 - variant_cr) / visitors_per_variant)
                mde_absolute = z_alpha * se + z_power * se
                
                # Calculate relative MDE based on the baseline conversion rate
                mde_relative = (mde_absolute / variant_cr) * 100
                
                # Append results for this week to the list
                results.append([week, visitors_per_variant, mde_relative])

            # Convert the list of results into a DataFrame
            df = pd.DataFrame(results, columns=['Week', 'Visitors / Variant', 'Relative MDE'])

            # Adjust formatting for better readability
            #df['Absolute MDE'] = df['Absolute MDE'].map(lambda x: f"{x:.2%}")
            df['Relative MDE'] = df['Relative MDE'].map(lambda x: f"{x:.2f}%")

            df = df.reset_index(drop=True)

            # Print the DataFrame
            st.write("")
            st.write("### MDE table")
            st.write("""
                This table tells you what the minimum effect is that you need to see in order to reach statistical significance 
                for the amount of weeks that your test has run. A relative MDE of < 5% is generally testworthy, 5-10% is debatable. For everything 
                above that, you should consider if the experiment will be able to achieve this effect and evaluate testworthiness.
            """)
            #st.write(df)
            # Convert DataFrame to HTML table without the index
            html_table = df.to_html(index=False)

            # Display the HTML table
            st.write(html_table, unsafe_allow_html=True)

if __name__ == "__main__":
    run()