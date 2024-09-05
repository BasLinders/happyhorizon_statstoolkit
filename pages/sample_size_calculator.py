import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

def run():
    st.set_page_config(
        page_title="Sample Size / MDE calculator",
        page_icon="🔢",
    )

    st.title("Sample Size Calculator")
    """
    This calculator provides you with the relative Minimum Detectable Effect (MDE) for your online experiment, over a period of 6 weeks.
    
    The total visitors and conversions will be distributed evenly across the variants. 
    The MDE is calculated for each week as traffic increases.
    
    Happy learning!
    """

    # Input: number of variants (arms)
    num_variants = st.number_input("How many variants will your experiment have (including control)?", min_value=2, step=1)

    # Input: total weekly visitors and conversions
    total_visitors = st.number_input("Amount of visitors per week:", min_value=0, step=1)
    total_conversions = st.number_input("Number of conversions per week:", min_value=0, step=1)

    # Trust (Power) and Risk (Confidence level) inputs
    trust = st.number_input("In %, enter the minimum trustworthiness (Power) (e.g. 80)", min_value=0, max_value=100, step=1)
    risk = st.number_input("In %, enter the desired confidence rate (e.g. 95)", min_value=0, max_value=100, step=1)
    tails = st.selectbox("Do you want to know if B is better than A, or also the other way around ('greater', 'two-sided')?", ('Greater', 'Two-sided'))

    st.write("")
    if st.button("Calculate Relative MDE"):
        if total_visitors <= 0 or total_conversions <= 0 or risk <= 0 or trust <= 0 or tails not in ['Greater', 'Two-sided']:
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)
        else:
            # Split the total visitors and conversions equally among the variants
            visitors_per_variant = total_visitors // num_variants
            conversions_per_variant = total_conversions // num_variants

            if visitors_per_variant <= 0 or conversions_per_variant <= 0:
                st.write("<span style='color: #ff6600;'>*Not enough visitors or conversions to distribute across variants</span>", unsafe_allow_html=True)
                return

            alpha = 1 - (risk / 100)
            power = trust / 100

            # Z-scores for confidence and power
            if tails == 'Two-sided':
                z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed test
            else:
                z_alpha = norm.ppf(1 - alpha)  # One-tailed test
            z_power = norm.ppf(power)

            # Baseline conversion rate (shared across all variants)
            baseline_rate = conversions_per_variant / visitors_per_variant

            # Initialize results list
            weeks = range(1, 7)  # For 6 weeks
            results = []

            # Weekly increments of visitors
            weekly_visitors_increase = np.ceil(visitors_per_variant) # Treating visitors as a constant 100% increase for each week

            for week in weeks:
                visitors_per_variant_weekly = int(weekly_visitors_increase * week)
                variant_cr = baseline_rate  # Treating conversion rate as a constant over weeks. 

                # Standard error calculation for MDE
                se = np.sqrt(2 * variant_cr * (1 - variant_cr) / visitors_per_variant_weekly)

                # Absolute and relative MDE calculation
                mde_absolute = (z_alpha + z_power) * se
                relative_mde = (mde_absolute / variant_cr) * 100

                # Append the result for this week
                results.append([week, visitors_per_variant_weekly, relative_mde])

            # Convert the list of results into a DataFrame
            df = pd.DataFrame(results, columns=['Week', 'Visitors / Variant', 'Relative MDE'])

            # Format the Relative MDE as a percentage for better readability
            df['Relative MDE'] = df['Relative MDE'].map(lambda x: f"{x:.2f}%")

            # Display the DataFrame as output
            st.write("")
            st.write("### Relative MDE over 6 Weeks")
            st.write("""
                This table shows the relative Minimum Detectable Effect (MDE) per week that you need to see in order to reach statistical significance 
                for the amount of weeks that your test has run. A relative MDE of < 5% is generally testworthy, 5-10% is debatable. For everything 
                above that, you should consider if the experiment will be able to achieve this effect and evaluate testworthiness.
            """)
            st.dataframe(df)

if __name__ == "__main__":
    run()