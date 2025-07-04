import streamlit as st
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import string
import concurrent.futures
import math

st.set_page_config(
    page_title="Frequentist Analysis",
    page_icon="",
)

def initialize_session_state():
    st.session_state.setdefault("num_variants", 2)
    st.session_state.setdefault("visitor_counts", [0] * st.session_state.num_variants)
    st.session_state.setdefault("variant_conversions", [0] * st.session_state.num_variants)
    st.session_state.setdefault("risk", 0)
    st.session_state.setdefault("tail", 'two-sided')

def display_intro():
    st.write("# Frequentist Analysis")
    st.markdown("""
                This calculator tests your data for statistical significance against the null-hypothesis (= there's no difference between 'A' and 'B'). 
                Input your experiment data and read the results at the bottom of the page.

                IMPORTANT: this calculator returns observed power values for the test. This statistic can be used for contextualizing the limitations in your test, albeit with a significant amount of caution.
                It is important to understand that observed power cannot be used to make any claims about the alternative hypothesis, since it is computed from a sample, not the total population.
                """)

def get_user_inputs():
    st.session_state.num_variants = st.number_input(
        "How many variants did your experiment have (including control)?",
        min_value=2, max_value=10, step=1, value=st.session_state.num_variants
    )

    if len(st.session_state.visitor_counts) != st.session_state.num_variants:
        st.session_state.visitor_counts = st.session_state.visitor_counts[:st.session_state.num_variants] + [0] * (st.session_state.num_variants - len(st.session_state.visitor_counts))
    if len(st.session_state.variant_conversions) != st.session_state.num_variants:
        st.session_state.variant_conversions = st.session_state.variant_conversions[:st.session_state.num_variants] + [0] * (st.session_state.num_variants - len(st.session_state.variant_conversions))

    num_variants = st.session_state.num_variants
    visitor_counts = st.session_state.visitor_counts
    variant_conversions = st.session_state.variant_conversions
    alphabet = string.ascii_uppercase

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Visitors")
    with col2:
        st.write("### Conversions")

    for i in range(num_variants):
        with col1:
            st.session_state.visitor_counts[i] = st.number_input(
                f"How many visitors did variant {alphabet[i]} have?",
                min_value=0, step=1, value=st.session_state.visitor_counts[i]
            )
        with col2:
            st.session_state.variant_conversions[i] = st.number_input(
                f"How many conversions did variant {alphabet[i]} have?",
                min_value=0, step=1, value=st.session_state.variant_conversions[i]
            )

    st.session_state.risk = st.number_input(
        "In %, how confident do you want to be in the results?",
        min_value=0, step=1, value=st.session_state.risk,
        help="Set the confidence level for which you want to test (enter 90, 95, etc)."
    )

    st.session_state.tail = st.selectbox(
        "Choose the test type:",
        options=['greater', 'two-sided', 'less'],
        index=['greater', 'two-sided', 'less'].index(st.session_state.tail),
        help="A one-sided test focuses on a change in one specific direction. Choose 'greater' if you only care whether the variant is significantly better than the control. Choose 'less' if you only care whether the variant is significantly worse than the control. For detecting a significant change in either direction (better or worse), choose 'two-sided'. P-value may be smaller in a one-sided test. Be aware that a one-sided test increases the risk of a Type I error."
    )

    return num_variants, visitor_counts, variant_conversions, st.session_state.risk, st.session_state.tail

def validate_inputs(visitor_counts, variant_conversions, num_variants):
    if any(v <= 0 for v in visitor_counts[:num_variants]):
        st.warning("Each variant must have at least one visitor.")
        return False
    if any(c > v for c, v in zip(variant_conversions[:num_variants], visitor_counts[:num_variants])):
        st.warning("Conversions cannot exceed visitors.")
        return False
    return True

def calculate_statistics(num_variants, visitor_counts, variant_conversions, risk, tail):
    alpha = 1 - (risk / 100)

    if num_variants >= 3:
        m = num_variants - 1  # Number of comparisons
        sidak_alpha = 1 - (1 - alpha)**(1 / m)
    else:
        sidak_alpha = alpha  # No correction needed if less than 3 variants

    st.write("### Please verify your input:")
    st.markdown(f"Chosen threshold for significance: {risk}%")
    st.markdown(f"Chosen test type: {'B is better than A' if tail == 'greater' else ('B is worse than A' if tail == 'less' else 'Two-sided')}.") # Added 'less', assumed other means two-sided

    for i in range(num_variants):
        st.markdown(f"Variant {string.ascii_uppercase[i]}: {visitor_counts[i]} visitors, {variant_conversions[i]} conversions")

    conversion_rates = [c / v for c, v in zip(variant_conversions, visitor_counts)]
    for i in range(num_variants):
        st.markdown(f" * Conversion Rate {string.ascii_uppercase[i]}: {conversion_rates[i] * 100:.2f}%")

    observed = np.array(visitor_counts)
    expected = np.array([sum(observed) / len(observed)] * len(observed))
    chi2, srm_p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    pooled_proportion = sum(variant_conversions) / sum(visitor_counts)
    se_list = [np.sqrt(pooled_proportion * (1 - pooled_proportion) / v) for v in visitor_counts]

    z_stats = [(conversion_rates[i] - conversion_rates[0]) / np.sqrt(se_list[i]**2 + se_list[0]**2) for i in range(1, num_variants)]

    p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_stats]

    if tail == 'greater':
        p_values = [p / 2 if z >= 0 else 1 - p / 2 for z, p in zip(z_stats, p_values)]
    elif tail == 'less':
        p_values = [p / 2 if z < 0 else 1 - p / 2 for z, p in zip(z_stats, p_values)]
    elif tail == 'two-sided':
        p_values = [p for p in p_values]

    st.write("")
    st.write("### Test statistics:")
    for i in range(1, num_variants):
        st.markdown(f" * Z-statistic for {string.ascii_uppercase[i]} vs {string.ascii_uppercase[0]}: {z_stats[i-1]:.4f}")
        st.markdown(f" * P-value for {string.ascii_uppercase[i]} vs {string.ascii_uppercase[0]}: {p_values[i-1]:.4f}")

    significant_results = [p <= sidak_alpha for p in p_values]

    if all(v > 1000 for v in visitor_counts):
        st.write("")
        st.write("\nUsing analytical approach to calculate observed power...\n")

        def analytical_power(cr_control, cr_variant, n_control, n_variant, alpha, tail):
            # Note: Using 'alpha' passed here (original overall alpha), not necessarily sidak_alpha
            se_unpooled = np.sqrt((cr_control * (1 - cr_control) / n_control) + (cr_variant * (1 - cr_variant) / n_variant))
            if se_unpooled == 0:
                return 1.0
            z_delta = abs(cr_control - cr_variant) / se_unpooled
            power = None
            
            if tail in ['greater', 'less']:
                z_alpha = stats.norm.ppf(1 - alpha)
                power = stats.norm.cdf(z_delta - z_alpha)
            else:
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                power = stats.norm.cdf(z_delta - z_alpha) + stats.norm.cdf(-z_delta - z_alpha)
            return power

        observed_powers = [analytical_power(conversion_rates[0], conversion_rates[i], visitor_counts[0], visitor_counts[i], alpha, tail) for i in range(1, num_variants)]
        for i in range(1, num_variants):
            st.markdown(f" * Observed power for {string.ascii_uppercase[i]} vs {string.ascii_uppercase[0]}: {observed_powers[i-1] * 100:.2f}%")

    else:
        st.write("")
        st.write("\nUsing bootstrapping to calculate power for more accuracy. Just a moment (running simulations)...\n")
        n_bootstraps = 10000 # Original value

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
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            # Adjust p-value based on tail
            if tail == 'greater':
                p_value /= 2
                p_value = 1 - p_value if z_stat < 0 else p_value
            elif tail == 'less':
                p_value_two_sided = p_value
                p_value = p_value_two_sided / 2 if z_stat < 0 else 1 - p_value_two_sided / 2
            elif tail == 'two-sided':
                p_value = p_value

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

        # prep and power call (uses overall alpha)
        data_controls = [np.concatenate([np.ones(c), np.zeros(v - c)]) for c, v in zip(variant_conversions, visitor_counts)]
        observed_powers = [bootstrap_power(data_controls[0], data_controls[i], alpha, tail, n_bootstraps=n_bootstraps) for i in range(1, num_variants)]
        for i in range(1, num_variants):
            st.markdown(f"   *Observed power for {string.ascii_uppercase[i]} vs {string.ascii_uppercase[0]}: {observed_powers[i-1] * 100:.2f}%")

    # return statement
    return conversion_rates, se_list, p_values, significant_results, observed_powers, srm_p_value, sidak_alpha

def visualize_results(conversion_rates, se_list, num_variants, significant_results, sidak_alpha):
    st.write("")
    st.write("### Probability Density Graph:")

    # Probability density graph
    plt.figure(figsize=(10, 6))

    # Define x_range dynamically based on the conversion rates
    all_means = np.array(conversion_rates)
    all_ses = np.array(se_list)
    # Calculate bounds based on +/- 4 standard errors from min/max means
    # Add small epsilon to SE for range calculation if SE is zero
    plot_min = np.min(all_means - 4 * np.maximum(all_ses, 1e-9))
    plot_max = np.max(all_means + 4 * np.maximum(all_ses, 1e-9))

    x_min = max(0, plot_min)
    x_max = min(1, plot_max)
    # Ensure x_min is strictly less than x_max
    if x_max <= x_min:
        x_max = x_min + 1e-6 # Add tiny offset if min/max are too close or equal
    x_range = np.linspace(x_min, x_max, 1000)


    # Define colors
    # Using a slightly larger, more distinct color palette
    colors = ['#808080', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Ensure control is grey
    colors[0] = '#808080' # Grey for control (index 0)

    shade_colors = {
        'better': '#90EE90', # lightgreen
        'worse': '#F08080'  # lightcoral
    }
    base_alpha = 0.9 # Alpha for lines
    shade_alpha = 0.3 # Alpha for fills (slightly reduced for better visibility)

    # Store plotted PDFs to avoid recalculation
    pdfs = []
    for i in range(num_variants):
        # Handle potential zero standard error
        se = max(se_list[i], 1e-9) # Avoid SE=0 for norm.pdf
        pdf = norm.pdf(x_range, conversion_rates[i], se)
        pdfs.append(pdf)

        variant_label = f'Variant {string.ascii_uppercase[i]}' if i > 0 else 'Control (A)'
        line_color = colors[i % len(colors)]

        plt.plot(x_range * 100, pdf, label=variant_label, color=line_color, alpha=base_alpha, linewidth=1.5)
        plt.axvline(conversion_rates[i] * 100, color=line_color, linestyle='--', alpha=base_alpha*0.8)

        # Add mean text label (adjust vertical position slightly)
        text_left_margin = 0.005
        plt.text(conversion_rates[i] * 100 + text_left_margin, 
                 plt.ylim()[1] * 0.03, # Lowered position
                 f' {string.ascii_uppercase[i]}: {conversion_rates[i]*100:.2f}%', # Label format
                 color=line_color, 
                 ha='left', 
                 rotation=90, 
                 va='bottom', 
                 fontsize=9
                 )

    # --- Shading & Probability Calculation ---
    control_cr = conversion_rates[0]
    control_se = max(se_list[0], 1e-9) # Avoid SE=0

    # Only loop through actual variants (index > 0) for comparison
    for i in range(1, num_variants):
        # Check the significance flag passed to the function for this variant vs control
        # The significance array index is i-1 because it doesn't include the control
        if significant_results[i - 1]:
            variant_cr = conversion_rates[i]
            variant_se = max(se_list[i], 1e-9) # Avoid SE=0
            pdf_variant = pdfs[i] # Use stored PDF

            is_better = variant_cr > control_cr
            shade_color = shade_colors['better'] if is_better else shade_colors['worse']
            variant_label_char = string.ascii_uppercase[i]
            control_label_char = string.ascii_uppercase[0]

            # --- Calculate probability P(Variant > Control) ---
            mean_diff = variant_cr - control_cr
            se_diff = math.sqrt(variant_se**2 + control_se**2)

            prob_variant_better = 0.5 # Default if se_diff is effectively zero
            if se_diff > 1e-9: # Check for non-zero se_diff before division
                z_score = mean_diff / se_diff
                prob_variant_better = norm.cdf(z_score) # Probability Variant i > Variant 0

            prob_control_better = 1 - prob_variant_better

            # --- Calculate bounds to exclude the relevant alpha tail ---
            if is_better:
                # Shade area where Variant > significance threshold
                lower_bound = norm.ppf(sidak_alpha, loc=variant_cr, scale=variant_se)
                fill_condition = (x_range >= lower_bound)
                bound_line_value = lower_bound * 100
            else:
                # Shade area where Variant < significance threshold
                upper_bound = norm.ppf(1 - sidak_alpha, loc=variant_cr, scale=variant_se)
                fill_condition = (x_range <= upper_bound)
                bound_line_value = upper_bound * 100

            # --- Shade the area EXCLUDING the alpha tail ---
            if prob_variant_better > prob_control_better:
                label_text = f'{variant_label_char} vs {control_label_char} (Significant)'
            elif prob_control_better > prob_variant_better:
                label_text = f'{control_label_char} vs {variant_label_char} (Significant)'
            else:
                label_text = ''
                
            plt.fill_between(x_range * 100, pdf_variant, 0,
                             where=fill_condition,
                             color=shade_color, alpha=shade_alpha,
                             label=label_text)
            # Format the probability text to display
            prob_text_display = f"P({variant_label_char}>{control_label_char}): {prob_variant_better*100:.1f}%"

            # --- Add boundary line ---
            plt.axvline(bound_line_value, color='grey', linestyle=':', linewidth=1, alpha=0.7)

            # --- *** FIX: Calculate mid_point_cr here *** ---
            mid_point_cr = (control_cr + variant_cr) / 2.0

            # --- Add probability text ---
            # Place text near the top, centered horizontally near the midpoint
            current_ylim = plt.ylim()
            y_pos_text = current_ylim[1] * 0.85 # Position near top
            plt.text(mid_point_cr * 100, 
                     y_pos_text, 
                     prob_text_display,
                     color='black', 
                     ha='center', 
                     va='center', 
                     fontsize=10,
                     #rotation = 90,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none')
                     )

            # Restore ylim in case text pushed it up slightly (though less likely with y_pos near top)
            plt.ylim(current_ylim)

    plt.xlabel('Conversion rate (%)')
    plt.ylabel('Probability density')
    plt.title('Comparison of Estimated Conversion Rate Distributions')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Move legend outside plot
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.3) # Add subtle grid
    #plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    st.pyplot(plt)
    plt.clf()

def summarize_results(conversion_rates, p_values, significant_results, observed_powers, num_variants, visitor_counts, srm_p_value, sidak_alpha, tail):
    st.write("### SRM Check")
    if srm_p_value > 0.01:
        st.write("This test is <span style='color: #009900; font-weight: 600;'>valid</span>. The distribution is as expected.", unsafe_allow_html=True)
    else:
        st.write("This test is <span style='color: #FF6600; font-weight: 600;'>invalid</span>: The distribution of traffic shows a statistically significant deviation from the expected values. Interpret the results with caution and check the distribution.", unsafe_allow_html=True)

    if num_variants >= 3:
        st.write("### Šidák Correction applied")
        st.write("The Šidák correction to solve for the Multiple Comparison Problem was applied due to 3 or more variants in the test.")

    st.write("## Results summary")
    def perform_superiority_test(i, alphabet, p_values, conversion_rates):
        if significant_results[i - 1]:
            st.write(f"### Test results for {alphabet[i]} vs {alphabet[0]}")
            st.markdown(f" * **Statistically significant result** for {alphabet[i]} with p-value: {p_values[i-1]:.4f}!")
            st.markdown(f" * **Observed power**: {observed_powers[i-1] * 100:.2f}%")
            st.markdown(f" * **Conversion rate change** for {alphabet[i]}: {((conversion_rates[i] - conversion_rates[0]) / conversion_rates[0]) * 100:.2f}%")
            if conversion_rates[i] > conversion_rates[0]:
                st.success(f"Variant **{alphabet[i]}** is a **winner**, congratulations!")
            else:
                st.warning(f"**Loss prevented** with variant **{alphabet[i]}**! Congratulations with this valuable insight.")

    def perform_non_inferiority_test(i, alphabet, p_values, conversion_rates, visitor_counts, alpha_noninf, non_inferiority_margin, tail):
        pooled_se = np.sqrt((conversion_rates[0] * (1 - conversion_rates[0]) / visitor_counts[0]) + (conversion_rates[i] * (1 - conversion_rates[i]) / visitor_counts[i]))
        z_stat_noninf = (conversion_rates[i] - conversion_rates[0] - non_inferiority_margin) / pooled_se
        p_value_noninf = stats.norm.cdf(z_stat_noninf)

        confidence_interval = stats.norm.interval(1 - alpha_noninf, loc=(conversion_rates[i] - conversion_rates[0]), scale=pooled_se)
        lower_bound, upper_bound = confidence_interval

        if p_values[i - 1] > sidak_alpha:
            st.write(f"### Test results for {alphabet[i]} vs {alphabet[0]}")
            st.markdown(f" * **Confidence interval for difference in conversion rates:** ({lower_bound:.4f}, {upper_bound:.4f})")
            st.markdown(f" * **Observed power:** {observed_powers[i-1] * 100:.2f}%")
            st.markdown(f" * **p-value:** {p_values[i-1]:.4f}")
            st.markdown(f" * **Conversion rate change for {alphabet[i]}:** {((conversion_rates[i] - conversion_rates[0]) / conversion_rates[0]) * 100:.2f}%")
            if tail == 'greater':
                st.markdown(f" * **P-value (non-inferiority test):** {p_value_noninf:.4f}")
                if p_value_noninf <= alpha_noninf:
                    st.success(f"Although the Z-test is not statistically significant (p = {p_values[i-1]:.4f}), "
                            f"the non-inferiority test suggests that {alphabet[i]} is **not significantly worse** than {alphabet[0]} "
                            f"within the predefined margin. This suggests that the variant may perform similarly to the control.")
                else:
                    st.warning(f"The Z-test is not statistically significant (p = {p_values[i-1]:.4f}), and the non-inferiority test "
                            f"does not provide sufficient evidence to conclude that {alphabet[i]} performs at least as well as {alphabet[0]}. "
                            f"More data may be needed.")
            else:
                st.info(f"The Z-test is not statistically significant (p = {p_values[i-1]:.4f}). There is no strong evidence of a difference, "
                        f"but the effect size remains uncertain. You may need to collect more data.")

    # Main logic
    alphabet = string.ascii_uppercase
    for i in range(1, num_variants):
        perform_superiority_test(i, alphabet, p_values, conversion_rates)

        alpha_noninf = 0.05
        non_inferiority_margin = 0.01
        perform_non_inferiority_test(i, alphabet, p_values, conversion_rates, visitor_counts, alpha_noninf, non_inferiority_margin, tail)

def run():
    initialize_session_state()
    display_intro()
    num_variants, visitor_counts, variant_conversions, risk, tail = get_user_inputs()
    risk = st.session_state.risk
    tail = st.session_state.tail

    st.write("")
    if st.button("Calculate my test results"):
        valid_inputs = validate_inputs(visitor_counts, variant_conversions, num_variants)

        if valid_inputs:
            conversion_rates, se_list, p_values, significant_results, observed_powers, srm_p_value, sidak_alpha = calculate_statistics(num_variants, visitor_counts, variant_conversions, risk, tail)
            visualize_results(conversion_rates, se_list, num_variants, significant_results, sidak_alpha)
            summarize_results(conversion_rates, p_values, significant_results, observed_powers, num_variants, visitor_counts, srm_p_value, sidak_alpha, tail)
        else:
            st.write("")
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()
