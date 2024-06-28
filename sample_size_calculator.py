try:
    import streamlit as st
    from scipy.stats import binom, norm, beta
    from statsmodels.stats.proportion import proportions_ztest
    import pandas as pd
    import numpy as np
    from scipy import optimize
    from decimal import Decimal, getcontext
    from statsmodels.stats.power import zt_ind_solve_power
    print("All libraries successfully imported!")
except ImportError as e:
    print(f"Failed to import libraries: {e}.")

# Inputs
baseline_visitors = st.number_input("Amount of visitors per week: ")
baseline_conversions = st.number_input("Number of conversions per week: ")
risk = st.number_input("In percentages, what is the risk you're willing to take (5, 10, 20, etc)? ")
trust = st.number_input("In percentages, how sure do you want to be that the effect exists (80, 90, etc)? ")
tails = st.number_input("Do you want to know if B is better than A, or also the other way around (enter '1-tailed' or '2-tailed')? ")
alpha = risk / 100
power = trust / 100

baseline_rate = float(round(baseline_conversions / baseline_visitors, 2))

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
#df['Variant CR'] = df['Variant CR'].map(lambda x: f"{x:.2%}")
df['Absolute MDE'] = df['Absolute MDE'].map(lambda x: f"{x:.2%}")
df['Relative MDE'] = df['Relative MDE'].map(lambda x: f"{x:.2f}%")

# Print the DataFrame
st.write("")
st.write(df.to_string(index=False))
