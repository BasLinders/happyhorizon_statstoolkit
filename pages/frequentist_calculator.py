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
    page_title="Frequentist Calculator",
    page_icon="",
)

def initialize_session_state():
    st.session_state.setdefault("num_variants", 2)
    st.session_state.setdefault("visitor_counts", [0] * st.session_state.num_variants)
    st.session_state.setdefault("variant_conversions", [0] * st.session_state.num_variants)
    st.session_state.setdefault("risk", 0)
    st.session_state.setdefault("tail", 'greater')

def display_intro():
    st.write("# Frequentist Calculator")
    st.markdown("""
                This calculator tests your data for statistical significance against the null-hypothesis (= there's no difference between 'A' and 'B'). 
                Input your experiment data and read the results at the bottom of the page.

                Happy learning!
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
        options=['greater', 'two-sided'],
        index=['greater', 'two-sided'].index(st.session_state.tail),
        help="A one-sided test ('greater') focuses only on improvement of B over A. For a change in either direction (better or worse), choose 'two-sided'."
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

    # Verify the data
    st.write("### Please verify your input:")
    st.markdown(f"Chosen threshold for significance: {risk}%")
    st.markdown(f"Chosen test type: {'B is better than A' if tail == 'greater' else 'B is worse than A'}.")

    for i in range(num_variants):
        st.markdown(f"Variant {string.ascii_uppercase[i]}: {visitor_counts[i]} visitors, {variant_conversions[i]} conversions")

    # Conversion rates
    conversion_rates = [c / v for c, v in zip(variant_conversions, visitor_counts)]
    for i in range(num_variants):
        st.markdown(f" * Conversion Rate {string.ascii_uppercase[i]}: {conversion_rates[i] * 100:.2f}%")

    # SRM check
    observed = np.array(visitor_counts)
    expected = np.array([sum(observed) / len(observed)] * len(observed))

    # Perform the chi-squared test for independence
    chi2, srm_p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    # Calculate pooled proportion and standard errors
    pooled_proportion = sum(variant_conversions) / sum(visitor_counts)
    se_list = [np.sqrt(pooled_proportion * (1 - pooled_proportion) / v) for v in visitor_counts]

    # Calculate the z-statistic for each variant against the control
    z_stats = [(conversion_rates[i] - conversion_rates[0]) / np.sqrt(se_list[i]**2 + se_list[0]**2) for i in range(1, num_variants)]

    # Calculate the p-values for each variant
    p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_stats]

    # Adjust for one-sided test if needed
    if tail == 'greater':
        p_values = [p / 2 if z >= 0 else 1 - p / 2 for z, p in zip(z_stats, p_values)]
    elif tail == 'two-sided':
        p_values = [p for p in p_values]

    st.write("")
    st.write("### Test statistics:")
    for i in range(1, num_variants):
        st.markdown(f" * Z-statistic for {string.ascii_uppercase[i]} vs {string.ascii_uppercase[0]}: {z_stats[i-1]:.4f}")
        st.markdown(f" * P-value for {string.ascii_uppercase[i]} vs {string.ascii_uppercase[0]}: {p_values[i-1]:.4f}")

    # Apply Sidak's correction to the p-values
    significant_results = [p <= sidak_alpha for p in p_values]

    # Power calculations
    if all(v > 1000 for v in visitor_counts):
        st.write("")
        st.write("\nUsing analytical approach to calculate observed power...\n")

        def analytical_power(cr_control, cr_variant, n_control, n_variant, alpha, tail):
            pooled_p = (cr_control * n_control + cr_variant * n_variant) / (n_control + n_variant)
            effect_size = abs(cr_control - cr_variant) / np.sqrt(pooled_p * (1 - pooled_p) * (1 / n_control + 1 / n_variant))
            if tail in ['greater', 'less']:
                z_alpha = stats.norm.ppf(1 - alpha)
                power = stats.norm.cdf(effect_size - z_alpha) if tail == 'greater' else stats.norm.cdf(effect_size + z_alpha)
            else:
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                power = stats.norm.cdf(effect_size - z_alpha) + stats.norm.cdf(-effect_size - z_alpha)
            return power

        observed_powers = [analytical_power(conversion_rates[0], conversion_rates[i], visitor_counts[0], visitor_counts[i], alpha, tail) for i in range(1, num_variants)]
        for i in range(1, num_variants):
            st.markdown(f" * Observed power for {string.ascii_uppercase[i]} vs {string.ascii_uppercase[0]}: {observed_powers[i-1] * 100:.2f}%")

    else:
        st.write("")
        st.write("\nUsing bootstrapping to calculate power for more accuracy. Just a moment (running simulations)...\n")
        n_bootstraps = 10000

        def bootstrap_sample(data_control, data_variant, alpha, tail):
            sample_control = np.random.choice(data_control, size=len(data_control), replace=True)
            sample_variant = np.random.choice(data_variant, size=len(data_variant), replace=True)
            pooled_p = (np.sum(sample_control) + np.sum(sample_variant)) / (len(sample_control) + len(sample_variant))
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / len(sample_control) + 1 / len(sample_variant)))
            z_stat = (np.mean(sample_variant) - np.mean(sample_control)) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            if tail == 'greater':
                p_value /= 2
                p_value = 1 - p_value if z_stat < 0 else p_value
            elif tail == 'two-sided':
                p_value = p_value
            return p_value < alpha

        def bootstrap_power(data_control, data_variant, alpha, tail, n_bootstraps=10000):
            significant_results = 0
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(bootstrap_sample, data_control, data_variant, alpha, tail) for _ in range(n_bootstraps)]
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        significant_results += 1
            return significant_results / n_bootstraps

        data_controls = [np.concatenate([np.ones(c), np.zeros(v - c)]) for c, v in zip(variant_conversions, visitor_counts)]
        observed_powers = [bootstrap_power(data_controls[0], data_controls[i], alpha, tail, n_bootstraps=n_bootstraps) for i in range(1, num_variants)]
        for i in range(1, num_variants):
            st.markdown(f"  *Observed power for {string.ascii_uppercase[i]} vs {string.ascii_uppercase[0]}: {observed_powers[i-1] * 100:.2f}%")

    return conversion_rates, se_list, p_values, significant_results, observed_powers, srm_p_value, sidak_alpha

def visualize_results(conversion_rates, se_list, num_variants, significant_results, sidak_alpha, risk):
    st.write("")
    st.write("### Probability Density Graph:")

    # Probability density graph
    plt.figure(figsize=(10, 6))

    # Define x_range dynamically based on the conversion rates
    all_means = np.array(conversion_rates)
    all_ses = np.array(se_list)
    # Calculate bounds based on +/- 4 standard errors from min/max means
    plot_min = np.min(all_means - 4 * all_ses)
    plot_max = np.max(all_means + 4 * all_ses)

    x_min = max(0, plot_min)
    x_max = min(1, plot_max)
    x_range = np.linspace(x_min, x_max, 1000)

    # Define colors
    colors = ['#A9A9A9', '#3498DB', '#9B59B6', '#E74C3C', '#1ABC9C', '#F39C12', '#2980B9', '#D35400', '#C0392B', '#7D3C98']

    colors[0] = '#AAAAAA' # Grey for control
    shade_colors = {
        'better': '#90EE90', # lightgreen
        'worse': '#F08080'  # lightcoral
    }
    base_alpha = 0.9 # Alpha for lines
    shade_alpha = 0.4 # Alpha for fills

    # Store plotted PDFs to avoid recalculation
    pdfs = []
    for i in range(num_variants):
        # Handle potential zero standard error if data is weird
        se = max(se_list[i], 1e-9) # Avoid SE=0
        pdf = norm.pdf(x_range, conversion_rates[i], se)
        pdfs.append(pdf)

        plt.plot(x_range * 100, pdf, label=f'Variant {string.ascii_uppercase[i]}', color=colors[i % len(colors)], alpha=base_alpha, linewidth=1.5)
        plt.axvline(conversion_rates[i] * 100, color=colors[i % len(colors)], linestyle='--', alpha=base_alpha*0.8)

        # Add mean text label (lowered to avoid probability text)
        plt.text(conversion_rates[i] * 100, plt.ylim()[1] * 0.05,
                 f'  {string.ascii_uppercase[i]}: {conversion_rates[i]*100:.2f}%', # Label format
                 color=colors[i % len(colors)], ha='left', rotation=90, va='bottom', fontsize=9) # Adjusted alignment

    # --- Shading & Probability Calculation ---
    control_cr = conversion_rates[0]
    control_se = max(se_list[0], 1e-9) # Avoid SE=0

    for i in range(1, num_variants): # Compare variants i > 0 against control (0)
        # Check the significance flag passed to the function for this variant vs control
        if significant_results[i - 1]:
            variant_cr = conversion_rates[i]
            variant_se = max(se_list[i], 1e-9) # Avoid SE=0
            pdf_variant = pdfs[i] # Use stored PDF

            is_better = variant_cr > control_cr
            shade_color = shade_colors['better'] if is_better else shade_colors['worse']

            # --- Calculate bounds to exclude the relevant alpha tail ---
            if is_better:
                lower_bound = norm.ppf(sidak_alpha, loc=variant_cr, scale=variant_se)
                fill_condition = (x_range >= lower_bound)
                # label_detail = f"(Lower {sidak_alpha*100:.1f}% tail excluded)" # Optional detail
            else:
                upper_bound = norm.ppf(1 - sidak_alpha, loc=variant_cr, scale=variant_se)
                fill_condition = (x_range <= upper_bound)
                # label_detail = f"(Upper {sidak_alpha*100:.1f}% tail excluded)" # Optional detail

            # --- Shade the area EXCLUDING the alpha tail ---
            label_text = f'{string.ascii_uppercase[i]} (Significant)'
            plt.fill_between(x_range * 100, pdf_variant, 0,
                             where=fill_condition,
                             color=shade_color, alpha=shade_alpha,
                             label=label_text)

            # --- Calculate and display probability P(Variant > Control) ---
            mean_diff = variant_cr - control_cr
            se_diff = math.sqrt(variant_se**2 + control_se**2)

            prob_variant_better = 0.5 # Default if se_diff is zero
            if se_diff > 1e-9: # Check for non-zero se_diff before division
                 # P(diff > 0) corresponds to the CDF of the standard normal distribution
                 # evaluated at (mean_diff / se_diff)
                 z_score = mean_diff / se_diff
                 prob_variant_better = norm.cdf(z_score) # Probability Variant i > Variant 0

            # Determine which probability to display (P(B>A) if B is better, P(A>B) if A is better)
            #prob_to_display = prob_variant_better if is_better else (1.0 - prob_variant_better)
            #prob_text = f"{prob_to_display*100:.0f}%"
            prob_text = f"{risk}%"

            # --- Add probability line and text ---
            mid_point_cr = (control_cr + variant_cr) / 2.0
            plt.axvline(mid_point_cr * 100, color='grey', linestyle=':', linewidth=1, alpha=0.7)

            # Place text near the top, centered horizontally near the midpoint line
            current_ylim = plt.ylim()
            y_pos_text = current_ylim[1] * 0.15
            plt.text(mid_point_cr * 100, y_pos_text, prob_text,
                     color='black', ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none')) # Add background box for readability

            # Restore ylim in case text pushed it up
            plt.ylim(current_ylim)

    plt.xlabel('Conversion rate (%)')
    plt.ylabel('Probability density')
    plt.title('Comparison of Estimated Conversion Rate Distributions')
    plt.legend()
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    #plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5) # Add subtle grid

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
            visualize_results(conversion_rates, se_list, num_variants, significant_results, sidak_alpha, risk)
            summarize_results(conversion_rates, p_values, significant_results, observed_powers, num_variants, visitor_counts, srm_p_value, sidak_alpha, tail)
        else:
            st.write("")
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()