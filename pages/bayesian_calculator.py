import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import beta

def run():
    st.set_page_config(
        page_title="Bayesian calculator",
        page_icon="🔢",
    )

    st.title("Bayesian Calculator")
    """
    This calculator outputs the probability of a variant to generate more conversions than the other. 
    Remember, you set the boundaries of success for the experiment; this calculator only helps you to translate it to numbers.

    It also shows the distribution of conversion rates by running a simulation and estimates the potential effect on revenue 
    over a chosen period after implementation with bayesian probability. Obviously, we make several statistical assumptions.

    Enter your experiment values below. Happy learning!
    """
    col1, col2 = st.columns(2)
    # Get visitor and conversion inputs with validation
    with col1:
        st.write("### Visitors")
        visitors_a = st.number_input("How many visitors does variant A have?", min_value=0, step=1)
        visitors_b = st.number_input("How many visitors does variant B have?", min_value=0, step=1)
    with col2:
        st.write("### Conversions")
        conversions_a = st.number_input("How many conversions does variant A have?", min_value=0, step=1)
        conversions_b = st.number_input("How many conversions does variant B have?", min_value=0, step=1)
    
    all_variant_conversions = [conversions_a, conversions_b]
    all_variant_visitors = [visitors_a, visitors_b]

    probability_winner = st.number_input("What is your minimum probability for a winner?", min_value=0.0, max_value=100.0, step=0.01)
    
    # Get projection period with validation
    st.write("")
    st.write("### Business case data")
    aov_a = st.number_input("What is the average order value of A? ", min_value=0.0, step=0.01)
    aov_b = st.number_input("What is the average order value of B? ", min_value=0.0, step=0.01)
    runtime_days = st.number_input("For how many days did your test run?", min_value=0, step=1)
    projection_period = st.number_input("Over how many days should we project the contribution in revenue?", min_value=0, step=1)
    conv_rate_a = conversions_a / visitors_a if visitors_a != 0 else 0
    conv_rate_b = conversions_b / visitors_b if visitors_b != 0 else 0
    uplift = (conv_rate_b - conv_rate_a) / conv_rate_a if conv_rate_a != 0 else 0

    valid_inputs = all(v > 0 for v in all_variant_visitors) and all(0 <= c <= v for c, v in zip(all_variant_conversions, all_variant_visitors))

    st.write("")
    if st.button("Calculate my test results"):
        if valid_inputs: 
            st.write("")
            st.write("Please verify your input:")
            st.write(f"Variant A: {visitors_a} visitors, {conversions_a} conversions, AOV: {aov_a}")
            st.write(f"Variant B: {visitors_b} visitors, {conversions_b} conversions, AOV: {aov_b}")
            st.write(f"Measured change in conversion rate: {uplift * 100:.2f}%")
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

                alpha_prior_business = conversions_a / visitors_a if visitors_a != 0 else 1
                beta_prior_business = (
                    alpha_prior_business * ((conv_rate_b - conv_rate_a) / conv_rate_a) * (1 / alpha_prior_business)
                    if alpha_prior_business != 0 and conv_rate_a != 0 else 1
                ) + alpha_prior_business

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

                # Function to simulate differences
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

                # Simulating the differences as percentage uplift
                diffs_percentage = simulate_differences(visitors_a, conversions_a, visitors_b, conversions_b) * 100
                conv_rate_a = conversions_a / visitors_a
                conv_rate_b = conversions_b / visitors_b
                observed_uplift = ((conv_rate_b - conv_rate_a) / conv_rate_a) * 100

                # Define a tighter range for the histogram
                mean_diff = np.mean(diffs_percentage)
                std_diff = np.std(diffs_percentage)
                range_min = mean_diff - 3 * std_diff
                range_max = mean_diff + 3 * std_diff

                # Plotting
                plt.figure(figsize=(14, 7))

                # Defining bins with a width of 2% for the histogram within the tighter range
                bin_width = 2  # 2% increments
                bins = np.arange(range_min, range_max, bin_width)

                # Plotting the histogram
                n, bins, patches = plt.hist(diffs_percentage, bins=bins, edgecolor='black', alpha=0.6)

                # Calculate the cumulative sum of the histogram frequencies
                cumsum = np.cumsum(n)
                total_samples = cumsum[-1]

                # Determine the number of samples favoring Variant A and Variant B
                num_a = np.sum(diffs_percentage < 0)
                num_b = total_samples - num_a

                # Determine the proportion of each bin to color
                for i in range(len(patches)):
                    if cumsum[i] < num_a:
                        patches[i].set_facecolor('lightcoral')  # Color for uplift related to Variant A
                    else:
                        patches[i].set_facecolor('lightgreen')  # Color for uplift related to Variant B

                # Setting dynamic x-ticks based on the bin range
                xticks = np.arange(range_min, range_max, bin_width)
                plt.xticks(xticks, [f'{tick:.0f}%' for tick in xticks], rotation=30)

                # Vertical line to visually provide uplift of variant
                if observed_uplift > 0:
                    line_observed_uplift = plt.axvline(x=observed_uplift, color='red', linestyle='--', 
                                linewidth=2, label=f'Observed Uplift B to A: {observed_uplift:.2f}%')
                else:
                    line_observed_uplift = plt.axvline(x=observed_uplift, color='red', linestyle='--', 
                                linewidth=2, label=f'Observed Drop B to A: {observed_uplift:.2f}%')

                # Creating dummy patches for the legend
                patch_a = mpatches.Patch(color='lightcoral', label='Variant A')
                patch_b = mpatches.Patch(color='lightgreen', label='Variant B')

                if observed_uplift > 0:
                    plt.title('Distribution of Simulated Conversion Rate Differences with Observed Uplift')
                else:
                    plt.title('Distribution of Simulated Conversion Rate Differences with Observed Drop')
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

                # risk assessment
                probability_a_better = 1 - probability_b_better

                # Calculate expected conversion rates for A and B
                alpha_post_a = alpha_prior_business + conversions_a
                beta_post_a = beta_prior_business + (visitors_a - conversions_a)
                expected_conv_rate_a = alpha_post_a / (alpha_post_a + beta_post_a)

                alpha_post_b = alpha_prior_business + conversions_b
                beta_post_b = beta_prior_business + (visitors_b - conversions_b)
                expected_conv_rate_b = alpha_post_b / (alpha_post_b + beta_post_b)

                # Simulate from posterior predictive distribution
                n_simulations = 10000
                samples_a = beta.rvs(alpha_post_a, beta_post_a, size=n_simulations)
                samples_b = beta.rvs(alpha_post_b, beta_post_b, size=n_simulations)

                # Calculate daily conversions for each simulation
                daily_conversions_a_samples = (samples_a * visitors_a) / runtime_days
                daily_conversions_b_samples = (samples_b * visitors_b) / runtime_days

                # Calculate expected daily conversions
                expected_daily_conversions_a = np.mean(daily_conversions_a_samples)
                expected_daily_conversions_b = np.mean(daily_conversions_b_samples)

                # Calculate credible intervals
                credible_interval_a = np.percentile(daily_conversions_a_samples, [2.5, 97.5])
                credible_interval_b = np.percentile(daily_conversions_b_samples, [2.5, 97.5])

                # Use the lower bounds of the credible intervals for conservative scenario
                lower_bound_a = credible_interval_a[0]
                lower_bound_b = credible_interval_b[0]

                # Conservative daily uplift
                if lower_bound_b > lower_bound_a:
                    conservative_daily_uplift = 0
                else:
                    conservative_daily_uplift = lower_bound_b - lower_bound_a

                # Calculate expected monetary uplift using conservative daily uplift
                expected_monetary_uplift_conservative = max(0, conservative_daily_uplift * aov_b * projection_period)

                # Print expected daily conversions and conservative uplift
                print(f"Expected daily conversions for A: {expected_daily_conversions_a:.2f}")
                print(f"Expected daily conversions for B: {expected_daily_conversions_b:.2f}")
                print(f"Conservative daily uplift: {conservative_daily_uplift:.2f}")
                print(f"Expected monetary uplift (conservative): {expected_monetary_uplift_conservative:.2f}")

                # Risk calculation using the original lower bound approach for comparison
                lower_bound_a_beta = beta.ppf(.01, alpha_post_a, beta_post_a) * visitors_a / runtime_days
                lower_bound_b_beta = beta.ppf(.01, alpha_post_b, beta_post_b) * visitors_b / runtime_days

                # Adjust expected monetary risk to account for the probability of A being better
                if probability_a_better > 0:
                    pessimistic_daily_diff = lower_bound_b_beta - lower_bound_a_beta
                    expected_monetary_risk = round((lower_bound_a_beta - lower_bound_b_beta) * aov_a * projection_period * probability_a_better, 2)
                else:
                    expected_monetary_risk = 0

                # Calculate improvement factor based on uplift in conversion rates
                improvement_factor = (expected_conv_rate_b - expected_conv_rate_a) / expected_conv_rate_a

                # Calculate optimistic daily difference over 180 days
                daily_uplift = expected_daily_conversions_b - expected_daily_conversions_a
                optimistic_daily_diff = daily_uplift * (1 + improvement_factor)

                # Optimistic monetary uplift
                optimistic_monetary_uplift = round(max(0, optimistic_daily_diff * aov_b * projection_period), 2)

                # Total contribution assuming this optimistic scenario
                total_contribution = round(optimistic_monetary_uplift + (-abs(expected_monetary_risk)), 2)

                # Construct dataframe with insights
                if probability_a_better > probability_b_better and lower_bound_b > lower_bound_a:
                    total_contribution = -abs(expected_monetary_risk)
                    optimistic_monetary_uplift = 0
                else:
                    total_contribution = total_contribution

                df = pd.DataFrame({
                    "B's chance to win": [probability_b_better * 100],
                    "Expected Monetary Uplift": [optimistic_monetary_uplift],
                    "Expected Monetary Risk": [-abs(expected_monetary_risk)],
                    "Expected Monetary Contribution": [total_contribution]
                })

                # Display results in Streamlit
                
                #st.write("## Results")
                #st.write("### Expected Daily Conversions")
                #st.write(f"A: {expected_daily_conversions_a}")
                #st.write(f"B: {expected_daily_conversions_b}")

                #st.write("### Lower Bounds")
                #st.write(f"A: {lower_bound_a}")
                #st.write(f"B: {lower_bound_b}")

                #st.write("### Priors")
                #st.write(f"Alpha Prior: {alpha_prior_business}")
                #st.write(f"Beta Prior: {beta_prior_business}")
                #st.write(f"Probability A better: {probability_a_better:.2%}")
                #st.write(f"Probability B better: {probability_b_better:.2%}")
                #st.write(f"Optimistic uplift: €{optimistic_monetary_uplift:.2f}")

                if round(probability_b_better * 100, 2) >= probability_winner:
                    bayesian_result = "<span style='color: green; font-weight: bold;'>Winner</span>"
                else:
                    bayesian_result = "<span style='color: red; font-weight: bold;'>Loss averted</span>"

                st.write("### Results Summary")
                st.write("")
                st.markdown(
                    f"Variant B has a {round(probability_b_better * 100, 2)}% chance to win with a relative change of {observed_uplift}%. "
                    f"Because your winning threshold was set to {probability_winner}%, this experiment is a {bayesian_result}.",
                    unsafe_allow_html=True
                )
                st.write("")
                st.write("The table below shows the contribution to the revenue over the projection period, with the AOVs as constants. This is purely a measurement for potential impact - no guarantee!")
                st.write("")
                st.dataframe(df)
        else:
            st.write("")
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields (business case is optional)</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()