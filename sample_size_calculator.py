import streamlit as st
from scipy.stats import norm
import pandas as pd
import numpy as np

st.title("Sample Size Calculator")

# Inputs
baseline_visitors = st.number_input("Amount of visitors per week:", min_value=0, step=1)
baseline_conversions = st.number_input("Number of conversions per week:", min_value=0, step=1)
risk = st.number_input("In percentages, what is the risk you're willing to take (5, 10, 20, etc)?", min_value=0.0, max_value=100.0, step=0.1)
trust = st.number_input("In percentages, how sure do you want to be that the effect exists (80, 90, etc)?", min_value=0.0, max_value=100.0, step=0.1)
tails = st.selectbox("Do you want to know if B is better than A, or also the other way around?", ('1-tailed', '2-tailed'))

# Ensure all inputs are provided and valid
if any([baseline_visitors <= 0, baseline_conversions <= 0, risk <= 0, trust <= 0, tails not in ['1-tailed', '2-tailed']]):
    st.write("Please enter all required inputs with valid values.")
else:
    alpha = risk / 100
    power = trust / 100

    # Calculate baseline conversion rate
    baseline_rate = baseline_conversions / baseline_visitors

    # Z-scores for confidence and power
    if tails == '2-tailed':
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
        
        # Sample size calculation adapted for two-tailed test, solving for MDE
        se = np.sqrt(2 * variant_cr * (1 - variant_cr) / visitors_per_variant)
        mde_absolute = z_alpha * se + z_power * se
        
        # Calculate relative MDE based on the baseline conversion rate
        mde_relative = (mde_absolute / variant_cr) * 100
        
        # Append results for this week to the list
        results.append([week, visitors_per_variant, mde_absolute, mde_relative])

    # Convert the list of results into a DataFrame
    df = pd.DataFrame(results, columns=['Week', 'Visitors / Variant', 'Absolute MDE', 'Relative MDE'])

    # Adjust formatting for better readability
    df['Absolute MDE'] = df['Absolute MDE'].map(lambda x: f"{x:.2%}")
    df['Relative MDE'] = df['Relative MDE'].map(lambda x: f"{x:.2f}%")

    # Print the DataFrame
    st.write("""
        This table tells you what the minimum effect is that you need to see in order to reach statistical significance 
        for the amount of weeks that your test has run. A MDE of < 5% is considered good, 5-10% is average. For everything 
        above that, you should consider if the test will be able to achieve this effect and evaluate testworthiness.
    """)
    st.write(df)