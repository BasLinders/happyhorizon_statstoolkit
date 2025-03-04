import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import beta

st.set_page_config(
    page_title="Bayesian calculator",
    page_icon="🔢",
)

def get_user_inputs():
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.visitors_a = st.number_input("How many visitors does variant A have?", min_value=0, step=1, value=st.session_state.get("visitors_a", 0))
        st.session_state.visitors_b = st.number_input("How many visitors does variant B have?", min_value=0, step=1, value=st.session_state.get("visitors_b", 0))
    with col2:
        st.session_state.conversions_a = st.number_input("How many conversions does variant A have?", min_value=0, step=1, value=st.session_state.get("conversions_a", 0))
        st.session_state.conversions_b = st.number_input("How many conversions does variant B have?", min_value=0, step=1, value=st.session_state.get("conversions_b", 0))
    
    st.session_state.probability_winner = st.number_input("Minimum probability for a winner?", min_value=0.0, max_value=100.0, step=0.01, value=st.session_state.get("probability_winner", 80.0), help="Enter the success percentage that determines if your test is a winner.")

    use_priors = st.checkbox("Use prior knowledge?", help="Take previous test data into account when evaluating this experiment.")
    if use_priors:
        expected_conversion = st.number_input("Expected Conversion Rate (%)", min_value=0, max_value=100, value=5)
        belief_strength = st.selectbox("Belief Strength", ["weak", "moderate", "strong"], index=1)
        alpha_prior, beta_prior = get_beta_priors(expected_conversion, belief_strength)
    else:
        alpha_prior, beta_prior = 1, 1  # Default uninformed priors
    
    st.write("")
    st.write("### Business case data")
    st.session_state.aov_a = st.number_input("What is the average order value of A? ", min_value=0.0, step=0.01, value=st.session_state.get("aov_a", 0.0))
    st.session_state.aov_b = st.number_input("What is the average order value of B? ", min_value=0.0, step=0.01, value=st.session_state.get("aov_b", 0.0))
    st.session_state.runtime_days = st.number_input("For how many days did your test run?", min_value=0, step=1, value=st.session_state.get("runtime_days", 0))
    
    return st.session_state.visitors_a, st.session_state.conversions_a, st.session_state.visitors_b, st.session_state.conversions_b, alpha_prior, beta_prior, st.session_state.probability_winner, st.session_state.aov_a, st.session_state.aov_b, st.session_state.runtime_days

def validate_inputs(visitors, conversions):
    if visitors is None or conversions is None:
        raise ValueError("Visitors and conversions cannot be zero")
    if not isinstance(visitors, int) or not isinstance(conversions, int):
        raise ValueError("Visitors and conversions must be whole numbers")
    if visitors < 0 or conversions < 0:
        raise ValueError("Visitors and conversions cannot be negative")
    if conversions > visitors:
        raise ValueError("Conversions cannot exceed the number of visitors")

def calculate_probability_b_better_and_samples(visitors_a, conversions_a, visitors_b, conversions_b, alpha_prior=1, beta_prior=1, num_samples=10000):
    alpha_post_a = alpha_prior + conversions_a
    beta_post_a = beta_prior + (visitors_a - conversions_a)
    alpha_post_b = alpha_prior + conversions_b
    beta_post_b = beta_prior + (visitors_b - conversions_b)

    samples_a = beta.rvs(alpha_post_a, beta_post_a, size=num_samples)
    samples_b = beta.rvs(alpha_post_b, beta_post_b, size=num_samples)

    probability_b_better = (samples_b > samples_a).mean()
    return probability_b_better, samples_a, samples_b

def get_beta_priors(expected_conversion_rate: int, belief_strength: str):
    if belief_strength not in ['weak', 'moderate', 'strong']:
        raise ValueError("belief_strength must be 'weak', 'moderate', or 'strong'")
    
    # Convert percentage to proportion
    conversion_rate = expected_conversion_rate / 100.0
    
    # Define prior weight based on belief strength
    prior_strengths = {
        'weak': 10,        # Light influence of prior data
        'moderate': 100,   # Medium influence
        'strong': 1000     # Heavy influence
    }
    
    k = prior_strengths[belief_strength]
    
    # Compute alpha and beta priors
    alpha_prior = conversion_rate * k
    beta_prior = (1 - conversion_rate) * k
    
    return int(alpha_prior), int(beta_prior)

def simulate_differences(visitors_a, conversions_a, visitors_b, conversions_b, alpha_prior=1, beta_prior=1, num_samples=10000):
    alpha_post_a = alpha_prior + conversions_a
    beta_post_a = beta_prior + (visitors_a - conversions_a)
    alpha_post_b = alpha_prior + conversions_b
    beta_post_b = beta_prior + (visitors_b - conversions_b)

    samples_a = beta.rvs(alpha_post_a, beta_post_a, size=num_samples)
    samples_b = beta.rvs(alpha_post_b, beta_post_b, size=num_samples)

    return (samples_b - samples_a) / samples_a

def plot_histogram(diffs_percentage, observed_uplift):
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
            patches[i]. set_facecolor('lightgreen')  # Color for uplift related to Variant B

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

def plot_probability_bar_chart(probability_b_better):
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

def perform_risk_assessment(visitors_a, conversions_a, visitors_b, conversions_b, aov_a, aov_b, runtime_days,
                           alpha_prior, beta_prior, probability_a_better, probability_b_better, projection_period=183):
    # Calculate expected conversion rates for A and B
    alpha_post_a = alpha_prior + conversions_a
    beta_post_a = beta_prior + (visitors_a - conversions_a)
    expected_conv_rate_a = alpha_post_a / (alpha_post_a + beta_post_a)

    alpha_post_b = alpha_prior + conversions_b
    beta_post_b = beta_prior + (visitors_b - conversions_b)
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

    # Risk calculation using the original lower bound approach for comparison
    lower_bound_a_beta = beta.ppf(.01, alpha_post_a, beta_post_a) * visitors_a / runtime_days
    lower_bound_b_beta = beta.ppf(.01, alpha_post_b, beta_post_b) * visitors_b / runtime_days

    # Adjust expected monetary risk to account for the probability of A being better
    if probability_a_better > 0:
        pessimistic_daily_diff = lower_bound_b_beta - lower_bound_a_beta
        expected_monetary_risk = round((lower_bound_a_beta - lower_bound_b_beta) * aov_a * projection_period * probability_a_better, 2)  # Use aov_a for variant A
    else:
        expected_monetary_risk = 0

    # Calculate improvement factor based on uplift in conversion rates
    improvement_factor = (expected_conv_rate_b - expected_conv_rate_a) / expected_conv_rate_a if expected_conv_rate_a > 0 else np.nan # handle edge cases

    # Calculate optimistic daily difference over 180 days
    daily_uplift = expected_daily_conversions_b - expected_daily_conversions_a
    optimistic_daily_diff = daily_uplift * (1 + improvement_factor)

    # Optimistic monetary uplift (use aov_b for variant B)
    optimistic_monetary_uplift = round(max(0, optimistic_daily_diff * aov_b * projection_period), 2)  # Use aov_b for variant B

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

    return df

def display_results(probability_b_better, observed_uplift, probability_winner, aov_a, aov_b, runtime_days, df=None):
    probability_a_better = 1 - probability_b_better

    if round(probability_b_better * 100, 2) >= probability_winner:
        bayesian_result = "a <span style='color: green; font-weight: bold;'>winner</span>"
    elif round(probability_a_better * 100, 2) >= probability_winner:
        bayesian_result = "a <span style='color: red; font-weight: bold;'>loss averted</span>"
    else:
        bayesian_result = "<span style='color: black; font-weight: bold;'>inconclusive</span>. There is no real effect to be found, or you need to collect more data"

    st.write("### Results Summary")
    st.write("")
    st.markdown(
        f"Variant B has a {round(probability_b_better * 100, 2)}% chance to win with a relative change of {round(observed_uplift, 2)}%. "
        f"Because your winning threshold was set to {int(probability_winner)}%, this experiment is {bayesian_result}.",
        unsafe_allow_html=True
    )

    # Check if business case data is valid
    if aov_a > 0 and aov_b > 0 and runtime_days > 0:
        if df is not None:
            st.write("")
            st.write("""
                    The table below shows the potential contribution to the revenue over a period of 6 months, with the AOVs as constants.
                    Please note: on smaller data sets of < 1000 conversions, the simulation might find more extreme values for a variant.
                    This could lead to inflated contributions; interpret with care. This table is purely a measurement for potential impact - no guarantee!
                    """)
            st.write("")
            st.dataframe(df)
    else:
        st.write("")
        st.warning("Business case data is missing or incomplete. Skipping monetary calculations.")

def run():
    st.title("Bayesian Calculator")
    """
    This calculator outputs the probability of a variant to generate more conversions than the other.
    Remember, you set the boundaries of success for the experiment; this calculator only helps you to translate it to numbers.

    It also shows the distribution of conversion rates by running a simulation and estimates the potential effect on revenue
    over a chosen period after implementation with bayesian probability. Obviously, we make several statistical assumptions.

    You can choose to incorporate expectations from previous experiments to take into account for the current experiment analysis.

    Enter your experiment values below. Happy learning!
    """

    # Get all user inputs using the dedicated function
    (visitors_a, conversions_a, visitors_b, conversions_b,
     alpha_prior, beta_prior, probability_winner,
     aov_a, aov_b, runtime_days) = get_user_inputs()

    # Input Validation (using the existing function) - Keep this outside the button press
    all_variant_conversions = [conversions_a, conversions_b]
    all_variant_visitors = [visitors_a, visitors_b]
    valid_inputs = all(v > 0 for v in all_variant_visitors) and all(
        0 <= c <= v for c, v in zip(all_variant_conversions, all_variant_visitors))

    st.write("")
    if st.button("Calculate my test results"):
        if valid_inputs:
            st.write("")
            st.write("Please verify your input:")
            st.write(f"Variant A: {visitors_a} visitors, {conversions_a} conversions, AOV: {aov_a}")
            st.write(f"Variant B: {visitors_b} visitors, {conversions_b} conversions, AOV: {aov_b}")

            # Calculate conversion rates and uplift (handle potential division by zero)
            conv_rate_a = conversions_a / visitors_a if visitors_a > 0 else 0
            conv_rate_b = conversions_b / visitors_b if visitors_b > 0 else 0
            uplift = (conv_rate_b - conv_rate_a) / conv_rate_a if conv_rate_a > 0 else np.nan  # Use NaN for undefined uplift
            st.write(f"Measured change in conversion rate: {uplift * 100:.2f}%")
            st.write(f"Minimum chance to win: {probability_winner}%")
            st.write(f"Test runtime: {runtime_days} days")

            probability_threshold = probability_winner / 100

            try:
                validate_inputs(visitors_a, conversions_a)
                validate_inputs(visitors_b, conversions_b)

                # --- Bayesian Calculations ---
                probability_b_better, samples_a, samples_b = calculate_probability_b_better_and_samples(
                    visitors_a, conversions_a, visitors_b, conversions_b, alpha_prior, beta_prior
                )

                # --- Output and Visualizations ---
                
                # Simulating the differences as percentage uplift
                diffs_percentage = simulate_differences(visitors_a, conversions_a, visitors_b, conversions_b, alpha_prior, beta_prior) * 100
                observed_uplift = uplift * 100

                plot_histogram(diffs_percentage, observed_uplift)
                plot_probability_bar_chart(probability_b_better)

                # --- Business Case (Risk Assessment) ---
                if aov_a > 0 and aov_b > 0 and runtime_days > 0:
                     # Use a consistent projection period (get from session state)
                    #st.session_state.projection_period = st.slider("Projection Period (days)", min_value=1, max_value=730, value=183, step=1)
                    #projection_period = st.session_state.projection_period
                    projection_period = 183

                    df = perform_risk_assessment(visitors_a, conversions_a, visitors_b, conversions_b,
                                                aov_a, aov_b, runtime_days, alpha_prior, beta_prior,
                                                1 - probability_b_better, probability_b_better, projection_period)
                else:
                    df = None

                display_results(probability_b_better, observed_uplift, probability_winner, aov_a, aov_b, runtime_days, df)

            except ValueError as e:
                st.write(f"Input Error: {e}")  # Handle validation errors

        else:
            st.write("")
            st.write(
                "<span style='color: #ff6600;'>*Please enter valid inputs for visitors and conversions (business case data is optional).</span>",
                unsafe_allow_html=True)

if __name__ == "__main__":
    run()