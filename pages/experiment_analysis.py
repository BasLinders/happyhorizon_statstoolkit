import streamlit as st
import pandas as pd
import numpy as np
import string
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.stats import beta, norm, chisquare
import math

st.set_page_config(
    page_title="Experiment Analysis",
    page_icon="🔢",
)

def initialize_session_state():
    st.session_state.setdefault("num_variants", 2)
    num_variants = st.session_state.num_variants
    st.session_state.setdefault("visitor_counts", [0] * num_variants)
    st.session_state.setdefault("conversion_counts", [0] * num_variants)
    st.session_state.setdefault("aovs", [0.0] * num_variants)
    st.session_state.setdefault("confidence_level", 95)
    st.session_state.setdefault("tail", 'two-sided')
    st.session_state.setdefault("probability_winner", 80.0)
    st.session_state.setdefault("runtime_days", 0)

# -- Data input functions
def get_bayesian_inputs():
    st.session_state.num_variants = st.number_input(
        "How many variants did your experiment have (including control)?",
        min_value=2, max_value=10, step=1,
        value=st.session_state.num_variants,
        key="bayesian_num_variants"
    )

    num_variants = st.session_state.num_variants

    # Ensure that the lists in the session state have the correct length
    for key, default_value in [("visitor_counts", 0), ("conversion_counts", 0), ("aovs", 0.0)]:
        if len(st.session_state[key]) != num_variants:
            current_len = len(st.session_state[key])
            if num_variants > current_len:
                st.session_state[key].extend([default_value] * (num_variants - current_len))
            else:
                st.session_state[key] = st.session_state[key][:num_variants]

    alphabet = string.ascii_uppercase
    st.write("---")

    # Create two columns for visitors and conversions
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Visitors")
    with col2:
        st.write("#### Conversions")

    # Add the input fields for each variant to the respective columns
    for i in range(num_variants):
        with col1:
            st.session_state.visitor_counts[i] = st.number_input(
                f"Visitors for variant {alphabet[i]}",
                min_value=0, step=1,
                value=st.session_state.visitor_counts[i],
                key=f"b_visitors_{i}",
                label_visibility="visible"
            )
        with col2:
            st.session_state.conversion_counts[i] = st.number_input(
                f"Conversions for variant {alphabet[i]}",
                min_value=0, step=1,
                value=st.session_state.conversion_counts[i],
                key=f"b_conversions_{i}",
                label_visibility="visible"
            )
            
    st.write("---")
    st.write("#### Average Order Value (€)")

    # Create columns for the AOV inputs. The number of columns is equal to the number of variants.
    aov_cols = st.columns(num_variants)
    for i, col in enumerate(aov_cols):
         with col:
            st.session_state.aovs[i] = st.number_input(
                f"Variant {alphabet[i]}",
                min_value=0.0, step=0.01,
                value=st.session_state.aovs[i],
                key=f"b_aov_{i}"
            )

    st.write("---")
    st.write("### General Test Settings")

    st.session_state.probability_winner = st.number_input(
        "Minimum probability for a winner?",
        min_value=0.0, max_value=100.0, step=0.01,
        value=st.session_state.probability_winner,
        help="Enter the success rate that determines if your test has a winner."
    )
    
    st.session_state.runtime_days = st.number_input(
        "How many days did your test run?",
        min_value=0, step=1, 
        value=st.session_state.runtime_days
    )

    use_priors = st.checkbox("Use adjusted priors?", help="Take into account previous test data when evaluating this experiment.")
    if use_priors:
        st.write("##### Prior Beliefs")
        col1, col2 = st.columns(2)
        with col1:
            expected_sample_size = st.number_input("What is the total expected sample size of the experiment?", min_value=1000, step=1, value=10000)
        with col2:
            expected_conversion = st.number_input("Expected Conversion Rate (%)", min_value=0.01, max_value=100.00, step=0.01, value=5.0)

        belief_strength = st.selectbox("Strength of Belief", ["weak", "moderate", "strong"], index=1, help="Indicate how strong your belief is in the expected conversion rate.")
        alpha_prior, beta_prior = get_beta_priors(expected_conversion, belief_strength, expected_sample_size)
    else:
        alpha_prior, beta_prior = 1.0, 1.0

    return (
        st.session_state.visitor_counts,
        st.session_state.conversion_counts,
        st.session_state.aovs,
        alpha_prior, beta_prior,
        st.session_state.probability_winner,
        st.session_state.runtime_days
    )
    
def get_frequentist_inputs():
    st.session_state.num_variants = st.number_input(
        "How many variants did your experiment have (including control)?",
        min_value=2, max_value=10, step=1,
        value=st.session_state.num_variants,
        key="frequentist_num_variants"
    )

    st.write("---")
    num_variants = st.session_state.num_variants

    for key, default_value in [("visitor_counts", 0), ("conversion_counts", 0), ("aovs", 0.0)]:
        if len(st.session_state[key]) != num_variants:
            current_len = len(st.session_state[key])
            if num_variants > current_len:
                st.session_state[key].extend([default_value] * (num_variants - current_len))
            else:
                st.session_state[key] = st.session_state[key][:num_variants]

    alphabet = string.ascii_uppercase
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Visitors")
    with col2:
        st.write("### Conversions")

    for i in range(st.session_state.num_variants):
        with col1:
            st.session_state.visitor_counts[i] = st.number_input(
                f"Visitors for variant {alphabet[i]}",
                min_value=0, step=1,
                value=st.session_state.visitor_counts[i],
                key=f"f_visitors_{i}"
            )
        with col2:
            st.session_state.conversion_counts[i] = st.number_input(
                f"Conversions for variant {alphabet[i]}",
                min_value=0, step=1,
                value=st.session_state.conversion_counts[i],
                key=f"f_conversions_{i}"
            )
    st.write("---")
    st.session_state.confidence_level = st.number_input(
        "In %, how confident do you want to be in the results?",
        min_value=0, step=1,
        value=st.session_state.get("confidence_level", 95),
        help="Set the confidence level for which you want to test (enter 90, 95, etc)."
    )
    
    return st.session_state.visitor_counts, st.session_state.conversion_counts, st.session_state.confidence_level

def validate_inputs(visitors, conversions, aovs=None):
    
    visitors_list = visitors if isinstance(visitors, list) else [visitors]
    conversions_list = conversions if isinstance(conversions, list) else [conversions]
    
    aovs_list = []
    if aovs is not None:
        aovs_list = aovs if isinstance(aovs, list) else [aovs]
    
    for i in range(len(visitors_list)):
        v = visitors_list[i]
        c = conversions_list[i]
        variant_name = chr(65 + i)

        if not isinstance(v, int) or not isinstance(c, int):
            st.error(f"Error for Variant {variant_name}: Visitors and conversions must be whole numbers.")
            return False
        if v < 0 or c < 0:
            st.error(f"Error for Variant {variant_name}: Visitors and conversions cannot be negative.")
            return False
        if c > v:
            st.error(f"Error for Variant {variant_name}: The amount of conversions ({c}) cannot exceed the amount of visitors ({v}).")
            return False
        
        if aovs_list and i < len(aovs_list):
            a = aovs_list[i]
            if not isinstance(a, (int, float)) or a < 0:
                st.error(f"Error for Variant {variant_name}: AOV must be a non-negative number.")
                return False
            
    return True


# -- Bayesian helper functions --

def calculate_probabilities(visitor_counts, conversion_counts, alpha_prior=1, beta_prior=1, num_samples=10000, seed=42):
    np.random.seed(seed)
    num_variants = len(visitor_counts)
    
    all_samples = []
    for i in range(num_variants):
        alpha_post = alpha_prior + conversion_counts[i]
        beta_post = beta_prior + (visitor_counts[i] - conversion_counts[i])
        
        samples = beta.rvs(alpha_post, beta_post, size=num_samples)
        all_samples.append(samples)
        
    samples_matrix = np.array(all_samples)

    best_variant_indices = np.argmax(samples_matrix, axis=0)
    
    probabilities_to_be_best = []
    for i in range(num_variants):
        prob = (best_variant_indices == i).mean()
        probabilities_to_be_best.append(prob)
        
    return probabilities_to_be_best, samples_matrix

def get_beta_priors(expected_conversion_rate: float, belief_strength: str, expected_sample_size: int):
    expected_conversion_rate = expected_conversion_rate / 100
    if belief_strength not in ['weak', 'moderate', 'strong']:
        raise ValueError("belief_strength must be 'weak', 'moderate', or 'strong'")

    if not (0 <= expected_conversion_rate <= 1):
        raise ValueError("expected_conversion_rate must be between 0 and 1.")

    if expected_sample_size <= 0:
        raise ValueError("expected_sample_size must be a positive integer.")

    # Base prior strength adjustment based on sample size.
    # The larger the sample size, the weaker the prior should be.
    sample_size_factor = 1 / (1 + math.log(expected_sample_size)) if expected_sample_size > 1 else 1.0

    prior_strengths = {
        'weak': 10 * sample_size_factor,
        'moderate': 100 * sample_size_factor,
        'strong': 1000 * sample_size_factor,
    }

    k = prior_strengths[belief_strength]

    alpha_prior = expected_conversion_rate * k
    beta_prior = (1 - expected_conversion_rate) * k

    return alpha_prior, beta_prior

def simulate_uplift_distributions(visitor_counts, conversion_counts, alpha_prior=1, beta_prior=1, num_samples=20000, seed=42):
    np.random.seed(seed)
    num_variants = len(visitor_counts)

    all_samples = []
    for i in range(num_variants):
        alpha_post = alpha_prior + conversion_counts[i]
        beta_post = beta_prior + (visitor_counts[i] - conversion_counts[i])
        samples = beta.rvs(alpha_post, beta_post, size=num_samples)
        all_samples.append(samples)
    
    samples_matrix = np.array(all_samples)
    samples_control = samples_matrix[0]
    
    uplift_distributions = []
    for i in range(1, num_variants):
        samples_challenger = samples_matrix[i]

        uplift_distribution = (samples_challenger - samples_control) / (samples_control + 1e-9)
        
        uplift_distributions.append(uplift_distribution)
        
    return uplift_distributions

def plot_uplift_histograms(uplift_distributions, observed_uplifts):
    num_challengers = len(uplift_distributions)
    alphabet = string.ascii_uppercase

    fig, axes = plt.subplots(
        nrows=num_challengers, 
        ncols=1, 
        figsize=(14, 8 * num_challengers), 
        squeeze=False
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        diffs_percentage = uplift_distributions[i] * 100
        observed_uplift = observed_uplifts[i] * 100
        challenger_label = alphabet[i + 1]
        control_label = alphabet[0]
        
        def calculate_optimal_bins(data):
            n = len(data)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr == 0: return int(1 + np.log2(n))
            bin_width_fd = 2 * iqr * (n ** (-1/3))
            if bin_width_fd == 0: return int(1 + np.log2(n))
            return min(int(np.ceil((np.max(data) - np.min(data)) / bin_width_fd)), 200)

        num_bins = calculate_optimal_bins(diffs_percentage)
        
        n, bins, patches = ax.hist(diffs_percentage, bins=num_bins, edgecolor='black', alpha=0.6)
        for patch in patches:
            if patch.get_x() < 0:
                patch.set_facecolor('lightcoral')
            else:
                patch.set_facecolor('lightgreen')

        mean_diff = np.mean(diffs_percentage)
        std_diff = np.std(diffs_percentage)
        range_min, range_max = mean_diff - 3.5 * std_diff, mean_diff + 3.5 * std_diff
        ax.set_xlim(range_min, range_max)

        try:
            plot_width_inches = fig.get_size_inches()[0]
            range_width = range_max - range_min
            
            renderer = fig.canvas.get_renderer()
            
            sample_text = ax.text(0.5, 0.5, f'{range_min:.2f}%', transform=ax.transAxes, ha='center', va='center')
            text_bbox = sample_text.get_window_extent(renderer)
            text_height_pixels = text_bbox.height
            sample_text.remove()

            dpi = fig.dpi
            text_height_inches = text_height_pixels / dpi
            min_tick_spacing_inches = text_height_inches * 1.5
            
            num_ticks_inches = int(plot_width_inches / min_tick_spacing_inches) if min_tick_spacing_inches > 0 else 5
            num_ticks = max(min(num_ticks_inches, 10), 2)
            
            xticks = np.linspace(range_min, range_max, num_ticks)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f'{tick:.2f}%' for tick in xticks], rotation=45, ha='right')
        except Exception:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}%'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        line_label = f'Observed Uplift ({challenger_label} vs {control_label}): {observed_uplift:.2f}%'
        line_observed_uplift = ax.axvline(x=observed_uplift, color='red', linestyle='--', linewidth=2, label=line_label)
        
        patch_a = mpatches.Patch(color='lightcoral', label=f'{control_label} is beter')
        patch_b = mpatches.Patch(color='lightgreen', label=f'{challenger_label} is beter')
        
        ax.set_title(f'Distribution of Simulated Uplift: Variant {challenger_label} vs. Variant {control_label}')
        ax.set_xlabel('Percentage Uplift in Conversion Rate (%)')
        ax.set_ylabel('Frequency')
        ax.legend(handles=[line_observed_uplift, patch_a, patch_b])
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)
    plt.close(fig)

def plot_winner_probabilities_chart(probabilities_to_be_best):
    num_variants = len(probabilities_to_be_best)
    
    alphabet = string.ascii_uppercase
    variant_labels = [f"Variant {alphabet[i]}" for i in range(num_variants)]
    colormap = plt.colormaps.get('viridis')
    colors = colormap(np.linspace(0, 1, num_variants))
    
    fig_height = 2 + num_variants * 0.8
    plt.figure(figsize=(10, fig_height))
    
    bars = plt.barh(variant_labels, probabilities_to_be_best, color=colors, edgecolor='black', alpha=0.8)
    
    plt.xlabel('Chance for Variants to be the Best')
    plt.title('Chance per Variant to generate the most Conversions')
    plt.xlim(0, 1.05)
    plt.gca().invert_yaxis()

    for index, value in enumerate(probabilities_to_be_best):
        if value > 0.9:
            plt.text(value - 0.02, index, f"{value:.2%}", ha='right', va='center', color='white', fontweight='bold', fontsize=12)
        else:
            plt.text(value + 0.01, index, f"{value:.2%}", ha='left', va='center', color='black', fontsize=11)

    st.pyplot(plt)
    plt.close()

def perform_multi_variant_risk_assessment(
    visitor_counts, 
    conversion_counts, 
    aovs,
    probabilities_to_be_best,
    runtime_days,
    alpha_prior=1, 
    beta_prior=1, 
    projection_period=183, 
    seed=42
):

    np.random.seed(seed)
    num_variants = len(visitor_counts)
    if runtime_days == 0:
        return pd.DataFrame()

    n_simulations = 20000
    all_daily_conversion_samples = []
    for i in range(num_variants):
        alpha_post = alpha_prior + conversion_counts[i]
        beta_post = beta_prior + (visitor_counts[i] - conversion_counts[i])
        
        samples_cr = beta.rvs(alpha_post, beta_post, size=n_simulations)
        
        daily_samples = (samples_cr * visitor_counts[i]) / runtime_days
        all_daily_conversion_samples.append(daily_samples)

    control_samples = all_daily_conversion_samples[0]
    control_aov = aovs[0]
    
    results = []
    for i in range(1, num_variants):
        challenger_samples = all_daily_conversion_samples[i]
        challenger_aov = aovs[i]
        difference_samples = challenger_samples - control_samples
        
        # --- Calculate potential additional revenue (Uplift) ---
        positive_diffs = difference_samples[difference_samples > 0]
        expected_daily_gain = np.mean(positive_diffs) if len(positive_diffs) > 0 else 0
        prob_challenger_is_better = (difference_samples > 0).mean()
        prob_control_is_better = (difference_samples < 0).mean()
        expected_monetary_uplift = expected_daily_gain * challenger_aov * projection_period * prob_challenger_is_better
        
        negative_diffs = difference_samples[difference_samples < 0]
        expected_daily_loss = np.mean(negative_diffs) if len(negative_diffs) > 0 else 0 # Negative number
        # expected_monetary_risk = expected_daily_loss * control_aov * projection_period # * prob_control_is_better is included in the mean
        expected_monetary_risk = expected_daily_loss * control_aov * projection_period * prob_control_is_better
        total_contribution = expected_monetary_uplift + expected_monetary_risk

        results.append({
            "Variant": string.ascii_uppercase[i],
            "Chance to Beat Control": round((prob_challenger_is_better * 100), 2),
            "Chance to be Best Overall": round((probabilities_to_be_best[i] * 100), 2),
            "Expected Monetary Uplift": round(expected_monetary_uplift, 2),
            "Expected Monetary Risk": round(expected_monetary_risk, 2),
            "Expected Total Contribution": round(total_contribution, 2)
        })

    if not results:
        return pd.DataFrame(columns=["Variant", "Chance to Beat Control", "Chance to be Best Overall", "Expected Monetary Uplift", "Expected Monetary Risk", "Expected Total Contribution"])

    df = pd.DataFrame(results)
    return df

def display_results_per_variant(
    probabilities_to_be_best, 
    observed_uplifts, 
    probability_winner,
    aovs, 
    runtime_days, 
    df=None
):

    num_variants = len(probabilities_to_be_best)
    alphabet = string.ascii_uppercase

    st.write("### Results Summary")
    st.write("")

    for i in range(1, num_variants):
        challenger_index = i
        control_index = 0
        
        challenger_label = alphabet[challenger_index]
        
        probability_challenger_better = probabilities_to_be_best[challenger_index]
        probability_control_better = probabilities_to_be_best[control_index]
        observed_uplift_challenger = observed_uplifts[i - 1]

        if round(probability_challenger_better * 100, 2) >= probability_winner:
            bayesian_result = "a <span style='color: green; font-weight: bold;'>winner</span>"
        elif round(probability_control_better * 100, 2) >= probability_winner:
            bayesian_result = "a <span style='color: red; font-weight: bold;'>loss averted</span>"
        else:
            bayesian_result = "<span style='color: black; font-weight: bold;'>inconclusive</span>. There is no real effect to be found, or you need to collect more data"
        
        st.write(f"#### Variant {challenger_label} vs Control (A)")
        st.markdown(
            f"Variant {challenger_label} has a {round(probability_challenger_better * 100, 2)}% chance to win with a relative change of {round(observed_uplift_challenger, 2)}%. "
            f"Because your winning threshold was set to {int(probability_winner)}%, this experiment is {bayesian_result}.",
            unsafe_allow_html=True
        )
        
        if num_variants > 2:
            st.write("---")

    if all(aov > 0 for aov in aovs) and runtime_days > 0:
        if df is not None:
            st.write("#### Business Risk Assessment")
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


# -- Frequentist helper functions --

def calculate_frequentist_statistics(visitor_counts, conversion_counts, confidence_level, tail):
    # --- Input Validation & Setup ---
    if sum(visitor_counts) == 0 or any(v < 0 for v in visitor_counts):
        raise ValueError("Visitor counts must be positive and sum to a non-zero value.")
    
    num_variants = len(visitor_counts)
    alpha = 1 - (confidence_level / 100)
    
    # Šídák correction
    sidak_alpha = 1 - (1 - alpha)**(1 / (num_variants - 1)) if num_variants > 2 else alpha

    # --- Core calculations ---
    conversion_rates = [c / v if v > 0 else 0 for c, v in zip(conversion_counts, visitor_counts)]
    standard_errors = [np.sqrt(cr * (1 - cr) / v) if v > 0 else 0 for cr, v in zip(conversion_rates, visitor_counts)]

    # Confidence interval calculation for conversion rates
    z_critical = norm.ppf(1 - (alpha / 2))
    margins_of_error = [z_critical * se for se in standard_errors]
    confidence_intervals = [
        (cr - moe, cr + moe)
        for cr, moe in zip(conversion_rates, margins_of_error)
    ]

    lower_boundaries = [interval[0] for interval in confidence_intervals]
    upper_boundaries = [interval[1] for interval in confidence_intervals]

    lowest_interval = min(lower_boundaries)
    highest_interval = max(upper_boundaries)
    

    # Confidence interval for the difference
    confidence_intervals_diff = []

    for i in range(1, num_variants):
        diff_cr = conversion_rates[i] - conversion_rates[0]
        se_diff = np.sqrt(standard_errors[i]**2 + standard_errors[0]**2)
        moe_diff = z_critical * se_diff
        
        # Confidence interval for the difference
        ci_diff = (diff_cr - moe_diff, diff_cr + moe_diff)
        confidence_intervals_diff.append(ci_diff)

    # SRM Check
    observed = np.array(visitor_counts)
    expected = np.array([sum(observed) / num_variants] * num_variants)
    _, srm_p_value = chisquare(f_obs=observed, f_exp=expected)
    
    # Z-statistics
    pooled_proportion = sum(conversion_counts) / sum(visitor_counts)
    se_pooled_list = [np.sqrt(pooled_proportion * (1 - pooled_proportion) / v) if v > 0 else 0 for v in visitor_counts]
    z_stats = [
        (conversion_rates[i] - conversion_rates[0]) / np.sqrt(se_pooled_list[i]**2 + se_pooled_list[0]**2) if (se_pooled_list[i]**2 + se_pooled_list[0]**2) > 0 else 0
        for i in range(1, num_variants)
    ]

    # P-values
    if tail == 'greater':
        p_values = [1 - norm.cdf(z) for z in z_stats]
    elif tail == 'less':
        p_values = [norm.cdf(z) for z in z_stats]
    else: # 'two-sided'
        p_values = [2 * (1 - norm.cdf(abs(z))) for z in z_stats]

    significant_results = [p <= sidak_alpha for p in p_values]

    # --- Observed Power Analysis ---
    power_method_used = ""
    observed_powers = []
    if all(v > 1000 for v in visitor_counts):
        power_method_used = "Analytical"
        def analytical_power(cr_c, cr_v, n_c, n_v, corrected_alpha, t):
            # Note: Using 'alpha' passed here (original overall alpha), not necessarily sidak_alpha
            se_unpooled = np.sqrt((cr_c * (1 - cr_c) / n_c) + (cr_v * (1 - cr_v) / n_v))
            if se_unpooled == 0:
                return 1.0
            z_delta = abs(cr_c - cr_v) / se_unpooled
            power = None
            
            if t in ['greater', 'less']:
                z_alpha = norm.ppf(1 - corrected_alpha)
                power = norm.cdf(z_delta - z_alpha)
            else:
                z_alpha = norm.ppf(1 - corrected_alpha / 2)
                power = norm.cdf(z_delta - z_alpha) + norm.cdf(-z_delta - z_alpha)
                
            return power
            
        observed_powers = [analytical_power(conversion_rates[0], conversion_rates[i], visitor_counts[0], visitor_counts[i], sidak_alpha, tail) for i in range(1, num_variants)]
    else:
        power_method_used = "Bootstrap"
        def bootstrap_sample(data_control, data_variant, alpha, tail):
            # Using 'alpha' passed here (original overall alpha), not necessarily sidak_alpha
            sample_control = np.random.choice(data_control, size=len(data_control), replace=True)
            sample_variant = np.random.choice(data_variant, size=len(data_variant), replace=True)
            pooled_p = (np.sum(sample_control) + np.sum(sample_variant)) / (len(sample_control) + len(sample_variant))
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / len(sample_control) + 1 / len(sample_variant)))
            if se == 0:
                z_stat = 0
            else:
                z_stat = (np.mean(sample_variant) - np.mean(sample_control)) / se
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))

            # Adjust p-value based on tail
            if tail == 'greater':
                p_value = 1 - norm.cdf(z_stat)
            elif tail == 'less':
                p_value = norm.cdf(z_stat)
            else:
                p_value = 2 * (1 - norm.cdf(abs(z_stat)))

            return p_value < alpha

        # bootstrap_power definition
        def bootstrap_power(data_control, data_variant, alpha, tail, n_bootstraps=10000):
            # Note: Uses overall alpha based on original call
            significant_results = 0
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(bootstrap_sample, data_control, data_variant, alpha, tail) for _ in range(n_bootstraps)]
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        significant_results += 1
            return significant_results / n_bootstraps
            
        data_controls = [np.concatenate([np.ones(c), np.zeros(v - c)]) for c, v in zip(conversion_counts, visitor_counts)]
        observed_powers = [bootstrap_power(data_controls[0], data_controls[i], alpha, tail) for i in range(1, num_variants)]

    # --- Single dictionary ---
    results = {
        "num_variants": num_variants,
        "tail": tail,
        "confidence_intervals": confidence_intervals, # CI for each variant
        "confidence_intervals_diff": confidence_intervals_diff, # CI for the difference
        "conversion_rates": conversion_rates,
        "lowest boundary": lowest_interval,
        "highest boundary": highest_interval,
        "conversion_rates": conversion_rates,
        "standard_errors": standard_errors,
        "z_stats": z_stats,
        "p_values": p_values,
        "is_significant": significant_results,
        "observed_powers": observed_powers,
        "power_method": power_method_used,
        "srm_p_value": srm_p_value,
        "sidak_alpha": sidak_alpha,
        "alpha": alpha,
        "confidence_level": confidence_level
    }
    
    return results
        
def plot_conversion_distributions(results):
    if not results:
        st.warning("Cannot generate visualization because calculation results are missing.")
        return

    conversion_rates = results['conversion_rates']
    se_list = results['standard_errors']
    num_variants = results['num_variants']
    significant_results = results['is_significant']
    sidak_alpha = results['sidak_alpha']
    
    st.write("")
    st.write("### Probability Density Graph:")

    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Dynamic plot range ---
    all_means = np.array(conversion_rates)
    all_ses = np.array(se_list)
    plot_min = np.min(all_means - 4 * np.maximum(all_ses, 1e-9))
    plot_max = np.max(all_means + 4 * np.maximum(all_ses, 1e-9))
    x_min = max(0, plot_min)
    x_max = min(1, plot_max)
    if x_max <= x_min: x_max = x_min + 1e-6
    x_range = np.linspace(x_min, x_max, 1000)

    # --- Define color pallette and plot PDFs ---
    colors = ['#808080', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    shade_colors = {'better': '#90EE90', 'worse': '#F08080'}
    base_alpha = 0.9
    shade_alpha = 0.3
    
    pdfs = []
    for i in range(num_variants):
        se = max(se_list[i], 1e-9)
        pdf = norm.pdf(x_range, conversion_rates[i], se)
        pdfs.append(pdf)
        
        variant_label = f'Variant {string.ascii_uppercase[i]}' if i > 0 else 'Control (A)'
        line_color = colors[i % len(colors)]

        ax.plot(x_range * 100, pdf, label=variant_label, color=line_color, alpha=base_alpha, linewidth=1.5)
        ax.axvline(conversion_rates[i] * 100, color=line_color, linestyle='--', alpha=base_alpha*0.8)
        
        # HERSTELD: Exacte kopie van de originele plt.text aanroep voor de gemiddelden
        text_left_margin = 0.005
        ax.text(conversion_rates[i] * 100 + text_left_margin, 
                ax.get_ylim()[1] * 0.03,
                f' {string.ascii_uppercase[i]}: {conversion_rates[i]*100:.2f}%',
                color=line_color, 
                ha='left', 
                rotation=90, 
                va='bottom', 
                fontsize=9)

    # --- Shade areas when significant ---
    control_cr = conversion_rates[0]
    control_se = max(se_list[0], 1e-9)
    
    for i in range(1, num_variants):
        if significant_results[i - 1]:
            variant_cr = conversion_rates[i]
            variant_se = max(se_list[i], 1e-9)
            pdf_variant = pdfs[i]
            is_better = variant_cr > control_cr
            shade_color = shade_colors['better'] if is_better else shade_colors['worse']
            variant_label_char = string.ascii_uppercase[i]
            control_label_char = string.ascii_uppercase[0]

            mean_diff = variant_cr - control_cr
            se_diff = math.sqrt(variant_se**2 + control_se**2)
            prob_variant_better = 0.5
            if se_diff > 1e-9:
                z_score = mean_diff / se_diff
                prob_variant_better = norm.cdf(z_score)
            prob_control_better = 1 - prob_variant_better
            
            if is_better:
                lower_bound = norm.ppf(sidak_alpha, loc=variant_cr, scale=variant_se)
                fill_condition = (x_range >= lower_bound)
                bound_line_value = lower_bound * 100
            else:
                upper_bound = norm.ppf(1 - sidak_alpha, loc=variant_cr, scale=variant_se)
                fill_condition = (x_range <= upper_bound)
                bound_line_value = upper_bound * 100

            if prob_variant_better > prob_control_better:
                label_text = f'{variant_label_char} vs {control_label_char} (Significant)'
            elif prob_control_better > prob_variant_better:
                label_text = f'{control_label_char} vs {variant_label_char} (Significant)'
            else:
                label_text = ''
            
            ax.fill_between(x_range * 100, pdf_variant, 0, where=fill_condition,
                            color=shade_color, alpha=shade_alpha, label=label_text)
            
            prob_text_display = f"P({variant_label_char}>{control_label_char}): {prob_variant_better*100:.1f}%"

            ax.axvline(bound_line_value, color='grey', linestyle=':', linewidth=1, alpha=0.7)
            
            mid_point_cr = (control_cr + variant_cr) / 2.0
            
            current_ylim = ax.get_ylim()
            y_pos_text = current_ylim[1] * 0.85
            ax.text(mid_point_cr * 100, 
                    y_pos_text, 
                    prob_text_display,
                    color='black', 
                    ha='center', 
                    va='center', 
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))
            ax.set_ylim(current_ylim)

    ax.set_xlabel('Conversion rate (%)')
    ax.set_ylabel('Probability density')
    ax.set_title('Comparison of Estimated Conversion Rate Distributions')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.set_ylim(bottom=0)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    st.pyplot(fig)
    plt.close(fig)


def display_frequentist_summary(
    results, 
    visitor_counts, 
    conversion_counts,
    non_inferiority_margin=0.01,
    confidence_noninf=95
):

    if not results:
        st.error("Calculation results are missing, cannot display summary.")
        return

    num_variants = results['num_variants']
    srm_p_value = results['srm_p_value']
    is_significant = results['is_significant']
    p_values = results['p_values']
    observed_powers = results['observed_powers']
    conversion_rates = results['conversion_rates']
    sidak_alpha = results['sidak_alpha']
    alpha_unadjusted = results['alpha']
    tail = results['tail']
    alphabet = string.ascii_uppercase
    
    # --- SRM Check and Šidák Info ---
    st.write("### SRM Check")
    if srm_p_value > 0.01:
        st.write("This test is <span style='color: #009900; font-weight: 600;'>valid</span>. The distribution is as expected.", unsafe_allow_html=True)
    else:
        st.write("This test is <span style='color: #FF6600; font-weight: 600;'>invalid</span>: The distribution of traffic shows a statistically significant deviation...", unsafe_allow_html=True)

    if num_variants >= 3:
        st.write("### Šidák Correction applied")
        st.info(f"The Šidák correction was applied due to 3 or more variants in the test. The alpha threshold has been set to **{results['sidak_alpha']:.4f}** instead of {alpha_unadjusted:.4f}.")
    
    st.write("## Results summary")
    st.write("---")

    for i in range(1, num_variants):
        challenger_index_in_lists = i - 1

        st.write(f"### Test results for {alphabet[i]} vs {alphabet[0]}")

        # Show confidence intervals for each variant
        control_ci = results['confidence_intervals'][0]
        challenger_ci = results['confidence_intervals'][i]
        ci_difference = results['confidence_intervals_diff'][challenger_index_in_lists] 
        observed_diff = conversion_rates[i] - conversion_rates[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label=f"Conversion Rate Control ({alphabet[0]})",
                value=f"{results['conversion_rates'][0]*100:.2f}%",
                help=f"The {results['confidence_level']}% confidence interval is [{control_ci[0]*100:.2f}% - {control_ci[1]*100:.2f}%]"
            )
        with col2:
            st.metric(
                label=f"Conversion Rate Challenger ({alphabet[i]})",
                value=f"{results['conversion_rates'][i]*100:.2f}%",
                help=f"The {results['confidence_level']}% confidence interval is [{challenger_ci[0]*100:.2f}% - {challenger_ci[1]*100:.2f}%]"
            )
        with col3:
            st.metric(
                label=f"Uplift CI ({alphabet[i]} vs {alphabet[0]})",
                value=f"{observed_diff*100:+.2f}%", # The '+' forces a +/- sign
                help=f"The {results['confidence_level']}% confidence interval for the uplift is from {ci_difference[0]*100:+.2f}% to {ci_difference[1]*100:+.2f}%."
            )
        st.write("")
        
        # --- Superiority Test ---
        if is_significant[challenger_index_in_lists]:
            st.markdown(f" * **Statistically significant result** for {alphabet[i]} with p-value: {p_values[challenger_index_in_lists]:.4f}!")
            st.markdown(f" * **Observed power**: {observed_powers[challenger_index_in_lists] * 100:.2f}%")
            st.markdown(f" * **Conversion rate change** for {alphabet[i]}: {((conversion_rates[i] - conversion_rates[0]) / conversion_rates[0]) * 100:.2f}%")
            if conversion_rates[i] > conversion_rates[0]:
                st.success(f"Variant **{alphabet[i]}** is a **winner**, congratulations!")
            else:
                st.warning(f"**Loss averted** with variant **{alphabet[i]}**! Congratulations with this valuable insight.")
        
        # --- Non-inferiority Test ---
        else:
            st.markdown(f" * The Z-test is not statistically significant (p = {p_values[challenger_index_in_lists]:.4f}).")
            st.markdown(f" * **Observed power**: {observed_powers[challenger_index_in_lists] * 100:.2f}%")
            st.markdown(f" * **Conversion rate change for {alphabet[i]}:** {((conversion_rates[i] - conversion_rates[0]) / conversion_rates[0]) * 100:.2f}%")

            if tail == 'greater' or tail == 'two-sided':
                se_unpooled = np.sqrt(
                    (conversion_rates[0] * (1 - conversion_rates[0]) / visitor_counts[0]) + 
                    (conversion_rates[i] * (1 - conversion_rates[i]) / visitor_counts[i])
                )
                
                z_stat_noninf = (conversion_rates[i] - conversion_rates[0] + non_inferiority_margin) / se_unpooled
                p_value_noninf = 1 - norm.cdf(z_stat_noninf)
                alpha_noninf = 1 - (confidence_noninf / 100)

                st.markdown(f" * **P-value (non-inferiority test):** {p_value_noninf:.4f} (margin: {non_inferiority_margin*100:.1f}%)")
                
                if p_value_noninf <= alpha_noninf:
                    st.success(f"Although not a winner, the non-inferiority test suggests that {alphabet[i]} is **not significantly worse** than {alphabet[0]} within the predefined margin.")
                else:
                    st.warning(f"The non-inferiority test does not provide sufficient evidence to conclude that {alphabet[i]} performs at least as well as {alphabet[0]}.")
            else: # Voor 'less' tail
                 st.info(f"There is no strong evidence of a difference, and the effect size remains uncertain.")

# Main logic
def run():
    st.title("Experiment Analysis")
    st.markdown("""
    This app provides methods for Bayesian analysis and Frequentist analysis (z-test). Choose the appropriate method for your case.
    
    ### Bayesian features
    - Probability assessement
    - Simulations of uplifts
    - Business risk assessment
    
    ### Frequentist features
    - Assessment for statistical significance
    - Non-inferiority assessment for non-significant uplifts
    """)
    st.write("---")
    initialize_session_state()
    
    analysis_method = st.selectbox(
        "Choose your analysis method:",
        ("Bayesian Analysis", "Frequentist Analysis")
    )
    st.write("---")

    # ==============================================================================
    #                             BAYESIAN ANALYSIS FLOW
    # ==============================================================================
    if analysis_method == "Bayesian Analysis":
        st.header("Bayesian Analysis Inputs")
        
        (
            visitor_counts, 
            conversion_counts, 
            aovs, 
            alpha_prior, 
            beta_prior,
            probability_winner, 
            runtime_days
        ) = get_bayesian_inputs()

        st.write("")
        if st.button("Calculate Bayesian Results"):
            if validate_inputs(visitor_counts, conversion_counts, aovs):
                st.write("---")
                try:
                    # --- Calculations ---
                    cr_control = conversion_counts[0] / visitor_counts[0] if visitor_counts[0] > 0 else 0
                    observed_uplifts = [
                        ((conversion_counts[i] / visitor_counts[i]) - cr_control) / cr_control if cr_control > 0 and visitor_counts[i] > 0 else 0.0
                        for i in range(1, len(visitor_counts))
                    ]

                    probabilities_to_be_best, _ = calculate_probabilities(
                        visitor_counts, conversion_counts, alpha_prior, beta_prior
                    )
                    uplift_distributions = simulate_uplift_distributions(
                        visitor_counts, conversion_counts, alpha_prior, beta_prior
                    )
                    df_business = perform_multi_variant_risk_assessment(
                        visitor_counts, conversion_counts, aovs,
                        probabilities_to_be_best, runtime_days,
                        alpha_prior, beta_prior
                    )

                    # --- Visualizations and Results ---
                    plot_winner_probabilities_chart(probabilities_to_be_best)
                    plot_uplift_histograms(uplift_distributions, observed_uplifts)
                    display_results_per_variant(
                        probabilities_to_be_best, observed_uplifts,
                        probability_winner, aovs, runtime_days, df=df_business
                    )

                except Exception as e:
                    st.error(f"An error occurred during calculation: {e}")
            else:
                pass

    # ==============================================================================
    #                           FREQUENTIST ANALYSIS FLOW
    # ==============================================================================
    elif analysis_method == "Frequentist Analysis":
        st.header("Frequentist Analysis Inputs")
        
        visitor_counts, conversion_counts, confidence_level = get_frequentist_inputs()
        
        st.session_state.tail = st.radio(
            "Select the test hypothesis (tail):",
            ('two-sided', 'greater', 'less'),
            horizontal=True,
            help="'two-sided' (A != B), 'greater' (B > A), 'less' (B < A). Be aware that 'greater' and 'less' are directional tests, while 'two-sided' is non-directional. Real-world problems often require a two-sided test, but you can choose based on your hypothesis."
        )
        non_inferiority_margin = st.number_input(
            "Non-inferiority margin (absolute %)", 
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="Set the acceptable negative performance margin for non-significant results (e.g., 1.0 for -1%)."
        ) / 100

        st.write("")
        if st.button("Calculate Frequentist Results"):
            if validate_inputs(visitor_counts, conversion_counts):
                st.write("---")
                try:
                    # --- Calculations ---
                    with st.spinner("Analysis in progress..."):
                        test_results = calculate_frequentist_statistics(
                            visitor_counts,
                            conversion_counts,
                            confidence_level,
                            st.session_state.tail
                        )

                        # --- Visualization and Results ---
                        if test_results:
                            plot_conversion_distributions(test_results)
                            display_frequentist_summary(
                                test_results, 
                                visitor_counts, 
                                conversion_counts,
                                non_inferiority_margin=non_inferiority_margin
                            )
                except Exception as e:
                    st.error(f"An error occurred during calculation: {e}")
            else:
                pass
            
if __name__ == "__main__":
    run()
