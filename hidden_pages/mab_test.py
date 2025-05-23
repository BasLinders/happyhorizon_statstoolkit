import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import string

def run():
    #st.set_page_config(
    #    page_title="Bayesian calculator",
    #    page_icon="🔢",
    #)

    # Generate fields for variants
    variant_visitors = []
    variant_conversions = []
    variant_aov = []
    variant_cr = []
    variant_uplift = []
    alphabet = string.ascii_uppercase

    # Validation of visitors and conversions
    def validate_inputs(visitors, conversions):
        if visitors is None or conversions is None:
            raise ValueError("Visitors and conversions cannot be zero")
        if not isinstance(visitors, int) or not isinstance(conversions, int):
            raise ValueError("Visitors and conversions must be whole numbers")
        if visitors < 0 or conversions < 0:
            raise ValueError("Visitors and conversions cannot be negative")
        if conversions > visitors:
            raise ValueError("Conversions cannot exceed the number of visitors")

    # Probability calculation
    def calculate_probabilities_better(visitors, conversions, alpha_prior, beta_prior, num_samples=10000):
        alpha_post = alpha_prior + np.array(conversions)
        beta_post = beta_prior + (np.array(visitors) - np.array(conversions))
        samples = [beta.rvs(alpha_post[i], beta_post[i], size=num_samples) for i in range(len(visitors))]
        probabilities_better = np.zeros((len(visitors), len(visitors)))
        for i in range(len(visitors)):
            for j in range(len(visitors)):
                if i != j:
                    probabilities_better[i, j] = (samples[i] > samples[j]).mean()
        probabilities_better_than_all = probabilities_better.mean(axis=1)
        return probabilities_better_than_all, samples

    # Simulating differences in conversion 
    def simulate_differences(visitors_a, conversions_a, visitors_b, conversions_b, num_samples=10000):
        alpha_prior = 1
        beta_prior = 1
        alpha_post_a = alpha_prior + conversions_a
        beta_post_a = beta_prior + (visitors_a - conversions_a)
        alpha_post_b = alpha_prior + conversions_b
        beta_post_b = beta_prior + (visitors_b - conversions_b)
        samples_a = beta.rvs(alpha_post_a, beta_post_a, size=num_samples)
        samples_b = beta.rvs(alpha_post_b, beta_post_b, size=num_samples)
        return (samples_b - samples_a) / samples_a

    # Histogram when num_variants == 2
    def plot_histogram(diffs_percentage, observed_uplift):
        mean_diff = np.mean(diffs_percentage)
        std_diff = np.std(diffs_percentage)
        range_min = mean_diff - 3 * std_diff
        range_max = mean_diff + 3 * std_diff
        bin_width = 2
        bins = np.arange(range_min, range_max, bin_width)
        n, bins, patches = plt.hist(diffs_percentage, bins=bins, edgecolor='black', alpha=0.6)
        cumsum = np.cumsum(n)
        total_samples = cumsum[-1]
        num_a = np.sum(diffs_percentage < 0)
        num_b = total_samples - num_a
        for i in range(len(patches)):
            patches[i].set_facecolor('lightcoral' if cumsum[i] < num_a else 'lightgreen')
        xticks = np.arange(range_min, range_max, bin_width)
        plt.xticks(xticks, [f'{tick:.0f}%' for tick in xticks], rotation=30)
        plt.axvline(x=observed_uplift, color='red', linestyle='--', linewidth=2,
                    label=f'Observed {"Uplift" if observed_uplift > 0 else "Drop"} B to A: {observed_uplift:.2f}%')
        plt.title('Distribution of Simulated Conversion Rate Differences')
        plt.xlabel('Conversion rate (%)')
        plt.ylabel('Frequency')
        plt.legend(handles=[mpatches.Patch(color='lightcoral', label='Variant A'),
                            mpatches.Patch(color='lightgreen', label='Variant B')])
        st.pyplot(plt)
        plt.clf()

    # Bar chart for probability comparisons
    def plot_bar_chart(variants, probabilities):
        plt.figure(figsize=(14, 6))
        bars = plt.barh(variants, probabilities, color=plt.cm.tab10.colors[:len(variants)])
        plt.xlabel('Chance to win')
        plt.title('Probability for a variant to generate more conversions')
        plt.xlim(0, 1)
        for index, value in enumerate(probabilities):
            plt.text(value + (0.02 if value <= 0.9 else -0.05), index, f"{value:.2%}",
                     ha='left' if value <= 0.9 else 'right', va='center',
                     color='white' if value > 0.9 else 'black', fontweight='bold' if value > 0.9 else 'normal')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

    # Business risk calculation
    def calculate_business_risk(num_variants, variant_visitors, variant_conversions, variant_aov, 
                                alpha_prior_business, beta_prior_business, probability_better_than_all, 
                                runtime_days, projection_period, confidence_level=0.95):
        
        # Initialize lists with default values
        expected_conv_rates = [0] * num_variants
        expected_daily_conversions = [0] * num_variants
        daily_uplifts = [0] * num_variants
        conservative_monetary_uplifts = [0] * num_variants
        conservative_monetary_risks = [0] * num_variants
        lower_bounds = [0] * num_variants
        conservative_improvement_factors = [0] * num_variants
        total_contributions = [0] * num_variants

        # Calculate the confidence interval bounds for the control variant
        alpha = (1 - confidence_level) / 2
        lower_bound_a = beta.ppf(alpha, alpha_prior_business[0] + variant_conversions[0],
                                beta_prior_business[0] + (variant_visitors[0] - variant_conversions[0])) * variant_visitors[0] / runtime_days

        for i in range(num_variants):
            # Apply truncation or winsorization to limit the influence of outliers
            variant_conversions[i] = min(max(variant_conversions[i], np.percentile(variant_conversions, 5)), np.percentile(variant_conversions, 95))
            
            alpha_post = alpha_prior_business[i] + variant_conversions[i]
            beta_post = beta_prior_business[i] + (variant_visitors[i] - variant_conversions[i])
            
            expected_conv_rate = alpha_post / (alpha_post + beta_post)
            expected_conv_rates[i] = expected_conv_rate
            
            expected_daily_conversions[i] = round((expected_conv_rate * variant_visitors[i]) / runtime_days)
            
            if i > 0:  # Skip the control variant in this block
                daily_uplifts[i] = expected_daily_conversions[i] - expected_daily_conversions[0]
                
                # Calculate the lower and upper bounds of the confidence interval
                lower_bound = beta.ppf(alpha, alpha_post, beta_post) * variant_visitors[i] / runtime_days
                upper_bound = beta.ppf(1 - alpha, alpha_post, beta_post) * variant_visitors[i] / runtime_days
                
                # Use the lower bound for a conservative estimate of uplift
                conservative_daily_uplift = lower_bound - expected_daily_conversions[0]
                
                # Apply a cap to the improvement factor to avoid overestimation
                conservative_improvement_factors[i] = min((lower_bound - expected_conv_rates[0]) / expected_conv_rates[0], 0.2)  # cap at 20%
                
                conservative_monetary_uplifts[i] = round(max(0, conservative_daily_uplift * variant_aov[i] * projection_period * (1 + conservative_improvement_factors[i])), 2)
                
                # Use the lower bound of the confidence interval for risk calculation
                lower_bounds[i] = lower_bound
                
                # Adjust risk based on probability of success and cap extreme variability
                base_risk = -round(abs((lower_bound_a - lower_bound) * variant_aov[0] * projection_period), 2)
                adjusted_risk = base_risk * (1 - probability_better_than_all[i])
                if adjusted_risk < -0.2 * variant_aov[i] * projection_period:
                    adjusted_risk = -0.2 * variant_aov[i] * projection_period  # cap at 20% potential loss
                conservative_monetary_risks[i] = adjusted_risk

                # Calculate the total contribution more conservatively
                total_contributions[i] = round(conservative_monetary_uplifts[i] + conservative_monetary_risks[i], 2)
            else:
                # Set default values for control (i == 0)
                daily_uplifts[i] = 0
                conservative_monetary_uplifts[i] = 0
                lower_bounds[i] = 0
                conservative_monetary_risks[i] = 0
                conservative_improvement_factors[i] = 0
                total_contributions[i] = 0

        # Assert that all lists have the same length
        assert len(expected_conv_rates) == len(conservative_monetary_risks) == len(total_contributions), "Inconsistent lengths detected."

        results = {
            "Variant": [f"Variant {chr(i + ord('A'))}" for i in range(1, num_variants)],
            "Chance to win (%)": [round(prob * 100, 2) for prob in probability_better_than_all[1:]],
            "Conservative Monetary Uplift (€)": conservative_monetary_uplifts[1:],  # Skip the control variant for output
            "Expected Monetary Risk (€)": conservative_monetary_risks[1:],          # Skip the control variant for output
            "Expected Monetary Contribution (€)": total_contributions[1:]           # Skip the control variant for output
        }

        df = pd.DataFrame(results)
        return df

    st.title("Bayesian Calculator")
    st.markdown("""
    Please note: This calculator for Bayesian multiple arm analysis is IN DEVELOPMENT and may contain errors and inconsistencies.
    """)
    num_variants = st.number_input("How many variants did your experiment have (including control)?", min_value=2, max_value=10, step=1)
    variant_visitors = []
    variant_conversions = []
    variant_aov = []
    alphabet = string.ascii_uppercase

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Visitors")
    with col2:
        st.write("### Conversions")
    
    for i in range(num_variants):
        with col1:
            visitors = st.number_input(f"How many visitors did variant {alphabet[i]} have? ", min_value=0, step=1, key=f"visitors_{i}")
            variant_visitors.append(visitors)
        with col2:
            conversions = st.number_input(f"How many conversions did variant {alphabet[i]} have? ", min_value=0, step=1, key=f"conversions_{i}")
            variant_conversions.append(conversions)
    
    probability_winner = st.number_input("What is your minimum probability for a winner? ", min_value=0.0, max_value=100.0, step=1.0, format="%.2f")
    
    st.write("\n### Business case data (optional)")
    for i in range(num_variants):
        aov = st.number_input(f"What is the average order value of variant {alphabet[i]}? ", min_value=0.0, step=0.1, key=f"aov_{i}")
        variant_aov.append(aov)
        
    runtime_days = st.number_input("For how many days did your test run? ", min_value=0, step=1)
    projection_period = st.number_input("Over how many days should we project the contribution in revenue? ", min_value=0, step=1)
    
    for i in range(num_variants):
        cr = variant_conversions[i] / variant_visitors[i] if variant_visitors[i] != 0 else 0
        variant_cr.append(cr)
    
    baseline_cr = variant_cr[0]
    for i in range(1, num_variants):
        uplift = (variant_cr[i] / baseline_cr) - 1 if baseline_cr != 0 else 0
        variant_uplift.append(uplift)
    
    valid_inputs = all(v > 0 for v in variant_visitors) and all(0 <= c <= v for c, v in zip(variant_conversions, variant_visitors))
    if st.button("Calculate my test results"):
        if valid_inputs: 
            st.write("\nPlease verify your input:")
            for i in range(num_variants):
                st.write(f"**Variant {alphabet[i]}**: {variant_visitors[i]} visitors, {variant_conversions[i]} conversions, AOV: {variant_aov[i]}")
                if i > 0:
                    st.write(f"Measured change in conversion rate for {alphabet[i]} vs {alphabet[0]}: {variant_uplift[i-1] * 100:.2f}%")
            st.write(f"Minimum chance to win: {probability_winner}%") 
            st.write(f"Test runtime: {runtime_days} days, Projection period: {projection_period} days")
            probability_threshold = probability_winner / 100
            try:
                for i in range(num_variants):
                    validate_inputs(variant_visitors[i], variant_conversions[i])
            except ValueError as e:
                st.write(f"Input Error: {e}")
            else:
                alpha_prior, beta_prior = 1, 1
                probability_better_than_all, samples = calculate_probabilities_better(variant_visitors, variant_conversions, alpha_prior, beta_prior)
                
                for i in range(1, num_variants):
                    if probability_better_than_all[i] >= probability_threshold:
                        st.write(f"\nVariant {alphabet[i]} has a higher chance of performing better than Variant {alphabet[0]} with a probability of {probability_better_than_all[i]:.2%}.")
                    elif probability_better_than_all[i] < (1 - probability_threshold):
                        st.write(f"\nVariant {alphabet[0]} has a higher chance of performing better than Variant {alphabet[i]} with a probability of {1 - probability_better_than_all[i]:.2%}.")
                    else:
                        st.write(f"\nNeither variant has a meaningful impact for more conversions with a probability of Variant {alphabet[i]} being better at {probability_better_than_all[i]:.2%}.")
                if num_variants == 2:
                    diffs_percentage = simulate_differences(variant_visitors[0], variant_conversions[0], variant_visitors[1], variant_conversions[1]) * 100
                    conv_rate_a = variant_conversions[0] / variant_visitors[0]
                    conv_rate_b = variant_conversions[1] / variant_visitors[1]
                    observed_uplift = ((conv_rate_b - conv_rate_a) / conv_rate_a) * 100
                    plot_histogram(diffs_percentage, observed_uplift)

                # Check list comprehension for errors in plotting n bars + 1 when num_variants == 2
                variants = [f"Variant {chr(i + ord('A'))}" for i in range(num_variants)]
                plot_bar_chart(variants, probability_better_than_all)

                alpha_prior_business = []
                beta_prior_business = []
                alpha_prior_control = variant_conversions[0]
                beta_prior_control = variant_visitors[0] - variant_conversions[0]
                for i in range(num_variants):
                    if i == 0:
                        alpha_prior_new = alpha_prior_control
                        beta_prior_new = beta_prior_control
                    else:
                        alpha_prior_new = variant_conversions[i]
                        beta_prior_new = variant_visitors[i] - variant_conversions[i]
                        uplift = (variant_cr[i] - variant_cr[0]) / variant_cr[0] if variant_cr[0] != 0 else 0
                        alpha_prior_new = round(alpha_prior_control * (1 + uplift), 6)
                    alpha_prior_business.append(alpha_prior_new)
                    beta_prior_business.append(beta_prior_new)
                alpha_prior_business = np.array(alpha_prior_business)
                beta_prior_business = np.array(beta_prior_business)
                df = calculate_business_risk(num_variants, variant_visitors, variant_conversions, variant_aov, alpha_prior_business, beta_prior_business, 
                                             probability_better_than_all, runtime_days, projection_period)
                st.write("### Results Summary\n")
                st.write("The table below shows the contribution to the revenue over the projection period, with the AOVs as constants (meaning they are equal for every conversion count). "
                "The calculator uses a conservative approach where extreme values are avoided to make it more realistic for most situations. "
                "It doesn't completely account for higher variability in the estimates. This is purely a measurement for potential impact - no guarantee!\n")
                st.dataframe(df)
        else:
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields (business case is optional)</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()