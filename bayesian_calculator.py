import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pingouin as pg
from scipy.stats import beta

st.write("# Bayesian Calculator")

# Get visitor and conversion inputs with validation
visitors_a = st.number_input("How many visitors does variant A have?", min_value=0, step=1)
conversions_a = st.number_input("How many conversions does variant A have?", min_value=0, step=1)
visitors_b = st.number_input("How many visitors does variant B have?", min_value=0, step=1)
conversions_b = st.number_input("How many conversions does variant B have?", min_value=0, step=1)
probability_winner = st.number_input("What is your minimum probability for a winner?", min_value=0.0, max_value=100.0, step=0.01)

# Get average order values with validation
aov_a = st.number_input("What is the average order value of A? ", min_value=0.0, step=0.01)
aov_b = st.number_input("What is the average order value of B? ", min_value=0.0, step=0.01)

# Get projection period with validation
runtime_days = st.number_input("For how many days did your test run?", min_value=0, step=1)
projection_period = st.number_input("Over how many days should we project the contribution in revenue?", min_value=0, step=1)

st.write("")
st.write("Please verify your input:")
st.write(f"Variant A: {visitors_a} visitors, {conversions_a} conversions, AOV: {aov_a}")
st.write(f"Variant B: {visitors_b} visitors, {conversions_b} conversions, AOV: {aov_b}")
st.write(f"Minimum chance to win: {probability_winner}%")
st.write(f"Test runtime: {runtime_days} days, Projection period: {projection_period} days")

probability_threshold = probability_winner / 100

# Validation of data
def validate_inputs(visitors, conversions):
    if visitors is None or conversions is None:
        raise ValueError("Visitors and conversions cannot be zero")
    if not isinstance(visitors, int) or not isinstance(conversions, int):
        raise ValueError("Visitors and conversions must be whole numbers")
    if visitors < 0 or conversions < 0:
        raise ValueError("Visitors and conversions cannot be negative")
    if conversions > visitors:
        raise ValueError("Conversions cannot exceed the number of visitors")

try:
    validate_inputs(visitors_a, conversions_a)
    validate_inputs(visitors_b, conversions_b)

    if visitors_a == 0 or visitors_b == 0:
        raise ValueError("Number of visitors cannot be zero")
    
    conv_rate_a = conversions_a / visitors_a if visitors_a != 0 else 0
    conv_rate_b = conversions_b / visitors_b if visitors_b != 0 else 0
    uplift = (conv_rate_b - conv_rate_a) / conv_rate_a if conv_rate_a != 0 else 0

except ValueError as e:
    st.write(f"Input Error: {e}")
else:
    alpha_prior, beta_prior = 1, 1

    alpha_prior_business = conversions_a / visitors_a if visitors_a != 0 else 0
    beta_prior_business = (
        alpha_prior_business * ((conv_rate_b - conv_rate_a) / conv_rate_a) * (1 / alpha_prior_business)
        if alpha_prior_business != 0 and conv_rate_a != 0 else 0
    ) + alpha_prior_business

    # st.write(f"Business risk alpha prior: {alpha_prior_business}")
    # st.write(f"Business risk beta prior: {beta_prior_business}")
    st.write("")

    def calculate_probability_b_better_and_samples(visitors_a, conversions_a, visitors_b, conversions_b, num_samples=10000):
        alpha_post_a = alpha_prior + conversions_a
        beta_post_a = beta_prior + (visitors_a - conversions_a)
        alpha_post_b = alpha_prior + conversions_b
        beta_post_b = beta_prior + (visitors_b - conversions_b)

        samples_a = beta.rvs(alpha_post_a, beta_post_a, size=num_samples)
        samples_b = beta.rvs(alpha_post_b, beta_post_b, size=num_samples)

        probability_b_better = (samples_b > samples_a).mean()
        return probability_b_better, samples_a, samples_b

    probability_b_better, samples_a, samples_b = calculate_probability_b_better_and_samples(visitors_a, conversions_a, visitors_b, conversions_b)

    if probability_b_better >= probability_threshold:
        st.write(f"\nVariant B has a higher chance of performing better than Variant A with a probability of {probability_b_better:.2%}.")
    elif probability_b_better < (1 - probability_threshold):
        st.write(f"\nVariant A has a higher chance of performing better than Variant B with a probability of {1 - probability_b_better:.2%}.")
    else:
        st.write(f"\nNeither variant has a meaningful impact for more conversions with a probability of Variant B being better at {probability_b_better:.2%}.")

    def simulate_differences(visitors_a, conversions_a, visitors_b, conversions_b, num_samples=10000):
        alpha_post_a = alpha_prior + conversions_a
        beta_post_a = beta_prior + (visitors_a - conversions_a)
        alpha_post_b = alpha_prior + conversions_b
        beta_post_b = beta_prior + (visitors_b - conversions_b)

        samples_a = beta.rvs(alpha_post_a, beta_post_a, size=num_samples)
        samples_b = beta.rvs(alpha_post_b, beta_post_b, size=num_samples)
        return (samples_b - samples_a) / samples_a

    diffs_percentage = simulate_differences(visitors_a, conversions_a, visitors_b, conversions_b) * 100
    observed_uplift = ((conv_rate_b - conv_rate_a) / conv_rate_a) * 100

    plt.figure(figsize=(14, 7))
    bin_width = 2
    min_diff = np.amin(diffs_percentage)
    max_diff = np.amax(diffs_percentage)
    bins = np.arange(min_diff - bin_width, max_diff + bin_width, bin_width)
    n, bins, patches = plt.hist(diffs_percentage, bins=bins, edgecolor='black', alpha=0.6)

    for i in range(len(patches)):
        if bins[i] < observed_uplift:
            patches[i].set_facecolor('lightcoral')
        else:
            patches[i].set_facecolor('lightgreen')

    xticks = np.arange(min_diff - bin_width, max_diff + bin_width, bin_width)
    plt.xticks(xticks, [f'{tick:.0f}%' for tick in xticks], rotation=30)

    if observed_uplift > 0:
        line_observed_uplift = plt.axvline(x=observed_uplift, color='red', linestyle='--', linewidth=2, label=f'Observed Uplift B to A: {observed_uplift:.2f}%')
    else:
        line_observed_uplift = plt.axvline(x=observed_uplift, color='red', linestyle='--', linewidth=2, label=f'Observed Drop B to A: {observed_uplift:.2f}%')

    patch_a = mpatches.Patch(color='lightcoral', label='Most Frequent Conversion Rates in A')
    patch_b = mpatches.Patch(color='lightgreen', label='Most Frequent Conversion Rates in B')

    plt.title('Distribution of Simulated Conversion Rate Differences with Observed Uplift' if observed_uplift > 0 else 'Distribution of Simulated Conversion Rate Differences with Observed Drop')
    plt.xlabel('Conversion rate (%)')
    plt.ylabel('Frequency')
    plt.legend(handles=[line_observed_uplift, patch_a, patch_b])
    st.pyplot(plt)
    plt.clf()

    probability_a_better = 1 - probability_b_better
    probabilities = [probability_a_better, probability_b_better]
    variants = ['Variant A', 'Variant B']

    plt.figure(figsize=(10, 4))
    bars = plt.barh(variants, probabilities, color=['#E15759', '#76B7B2'])
    plt.xlabel('Chance to win')
    plt.title('Probability for a variant to generate more conversions')
    plt.xlim(0, 1)

    for index, value in enumerate(probabilities):
        if value > 0.9:
            plt.text(value - 0.05, index, f"{value:.2%}", ha='right', va='center', color='white', fontweight='bold')
        else:
            plt.text(value + 0.02, index, f"{value:.2%}", ha='left', va='center')

    st.pyplot(plt)
    plt.clf()

    if runtime_days <= 0:
        raise ValueError("Runtime days must be a positive integer")
    if projection_period <= 0:
        raise ValueError("Projection period must be a positive value")

    alpha_post_a = alpha_prior_business + conversions_a
    beta_post_a = beta_prior_business + (visitors_a - conversions_a)
    expected_conv_rate_a = alpha_post_a / (alpha_post_a + beta_post_a)

    alpha_post_b = alpha_prior_business + conversions_b
    beta_post_b = beta_prior_business + (visitors_b - conversions_b)
    expected_conv_rate_b = alpha_post_b / (alpha_post_b + beta_post_b)

    expected_daily_conversions_a = round((expected_conv_rate_a * visitors_a) / runtime_days)
    expected_daily_conversions_b = round((expected_conv_rate_b * visitors_b) / runtime_days)
    daily_uplift = expected_daily_conversions_b - expected_daily_conversions_a
    expected_monetary_uplift = daily_uplift * aov_b * projection_period

    lower_bound_a = beta.ppf(.01, alpha_post_a, beta_post_a) * visitors_a / runtime_days
    lower_bound_b = beta.ppf(.01, alpha_post_b, beta_post_b) * visitors_b / runtime_days

    if probability_a_better > 0:
        expected_monetary_risk = (lower_bound_a - lower_bound_b) * aov_a * projection_period * probability_a_better
    else:
        expected_monetary_risk = 0

    improvement_factor = (expected_conv_rate_b - expected_conv_rate_a) / expected_conv_rate_a
    optimistic_daily_diff = daily_uplift * (1 + improvement_factor)
    optimistic_monetary_uplift = optimistic_daily_diff * aov_b * projection_period
    total_contribution = optimistic_monetary_uplift + expected_monetary_risk

    # Create DataFrame and round the values to 2 decimal places
    df = pd.DataFrame({
        "B's chance to win (%)": [probability_b_better * 100],
        "Uplift (€)": [optimistic_monetary_uplift],
        "Risk (€)": [expected_monetary_risk],
        "Total Contribution (€)": [total_contribution]
    }).round(2)

    #st.write("## Expected Daily Conversions:")
    #st.write(f"Variant A: {expected_daily_conversions_a} conversions/day")
    #st.write(f"Variant B: {expected_daily_conversions_b} conversions/day")

    #st.write("## Lower Bounds:")
    #st.write(f"Variant A: {lower_bound_a:.2f} conversions/day")
    #st.write(f"Variant B: {lower_bound_b:.2f} conversions/day")

    #st.write("## Priors:")
    #st.write(f"Alpha prior for business risk: {alpha_prior_business}")
    #st.write(f"Beta prior for business risk: {beta_prior_business}")

    st.write("## Results:")
    st.write("Below is the summary of the outcomes including B's chance to win, potential uplift, risk, and total contribution over the specified projection period:")
    st.dataframe(df)
