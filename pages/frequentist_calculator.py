import streamlit as st
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import string
import concurrent.futures

def run():
    st.set_page_config(
        page_title="Frequentist Calculator",
        page_icon="🔢",
    )

    # Initialize session state for inputs
    st.session_state.setdefault("num_variants", 2)
    st.session_state.setdefault("visitor_counts", [0] * st.session_state.num_variants)
    st.session_state.setdefault("variant_conversions", [0] * st.session_state.num_variants)
    st.session_state.setdefault("risk", 0)
    st.session_state.setdefault("tail", 'Greater')

    st.write("# Frequentist Calculator")
    st.markdown("""
                This calculator tests your data for statistical significance against the null-hypothesis (= there's no difference between 'A' and 'B'). 
                Input your experiment data and read the results at the bottom of the page.

                Happy learning!
                """)

    # Number of variants input
    st.session_state.num_variants = st.number_input(
        "How many variants did your experiment have (including control)?",
        min_value=2, max_value=10, step=1, value=st.session_state.num_variants
    )

    # Adjust visitor_counts and variant_conversions to match num_variants without overwriting existing values
    if len(st.session_state.visitor_counts) != st.session_state.num_variants:
        st.session_state.visitor_counts = st.session_state.visitor_counts[:st.session_state.num_variants] + [0] * (st.session_state.num_variants - len(st.session_state.visitor_counts))
    if len(st.session_state.variant_conversions) != st.session_state.num_variants:
        st.session_state.variant_conversions = st.session_state.variant_conversions[:st.session_state.num_variants] + [0] * (st.session_state.num_variants - len(st.session_state.variant_conversions))

    # Assign local variables for convenience
    num_variants = st.session_state.num_variants
    visitor_counts = st.session_state.visitor_counts
    variant_conversions = st.session_state.variant_conversions
    alphabet = string.ascii_uppercase

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Visitors")
    with col2:
        st.write("### Conversions")

    # Update visitor_counts and variant_conversions for each variant
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

    # Confidence level and test type inputs
    st.session_state.risk = st.number_input(
        "In %, how confident do you want to be in the results (enter 90, 95, etc)?",
        min_value=0, step=1, value=st.session_state.risk
    )

    st.session_state.tail = st.selectbox(
        "Do you only want to know if B is better than A ('greater'), or if B is worse than A ('two-sided')?",
        options=['greater', 'two-sided'],
        index=['greater', 'two-sided'].index(st.session_state.tail)
    )

    # Validation check
    valid_inputs = all(v > 0 for v in visitor_counts[:num_variants]) and all(
        0 <= c <= v for c, v in zip(variant_conversions[:num_variants], visitor_counts[:num_variants])
    )

    st.write("")
    if st.button("Calculate my test results"):

        if valid_inputs:
            tail = st.session_state.tail
            risk = st.session_state.risk
            alpha = 1 - (risk / 100)

            # Apply Sidak's correction if there are 3 or more variants
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
                st.markdown(f"Variant {alphabet[i]}: {visitor_counts[i]} visitors, {variant_conversions[i]} conversions")

            # Conversion rates
            conversion_rates = [c / v for c, v in zip(variant_conversions, visitor_counts)]
            for i in range(num_variants):
                st.markdown(f" * Conversion Rate {alphabet[i]}: {conversion_rates[i] * 100:.2f}%")

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
                st.markdown(f" * Z-statistic for {alphabet[i]} vs {alphabet[0]}: {z_stats[i-1]:.4f}")
                st.markdown(f" * P-value for {alphabet[i]} vs {alphabet[0]}: {p_values[i-1]:.4f}")

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
                    st.markdown(f" * Observed power for {alphabet[i]} vs {alphabet[0]}: {observed_powers[i-1] * 100:.2f}%")

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
                    st.markdown(f"  *Observed power for {alphabet[i]} vs {alphabet[0]}: {observed_powers[i-1] * 100:.2f}%")

            st.write("")
            st.write("### Probability Density Graph:")

            # Probability density graph
            plt.figure(figsize=(10, 6))

            # Define x_range dynamically based on the conversion rates
            min_conversion_rate = min(conversion_rates) - 4 * max(se_list)
            max_conversion_rate = max(conversion_rates) + 4 * max(se_list)
            x_min = max(0, min_conversion_rate)
            x_max = min(1, max_conversion_rate)
            x_range = np.linspace(x_min, x_max, 1000)

            colors = ['#FF5733', '#3498DB', '#9B59B6', '#E74C3C', '#1ABC9C', '#F39C12', '#2980B9', '#D35400', '#C0392B', '#7D3C98']

            # Plot the probability density functions
            for i in range(num_variants):
                pdf = norm.pdf(x_range, conversion_rates[i], se_list[i])
                plt.plot(x_range * 100, pdf, label=f'Variant {alphabet[i]}', color=colors[i])
                plt.axvline(conversion_rates[i] * 100, color=colors[i], linestyle='--')
                plt.text(conversion_rates[i] * 100, plt.ylim()[1] * 0.50, f'Mean {alphabet[i]}', color=colors[i], ha='right', rotation=90, va='bottom')

                # Add shading for significant results
                if i > 0 and significant_results[i - 1]:
                    if conversion_rates[i] > conversion_rates[0]:
                        upper_critical_value = norm.ppf(1 - sidak_alpha, loc=conversion_rates[i], scale=se_list[i])
                        plt.fill_between(x_range * 100, pdf, where=(x_range * 100 >= upper_critical_value * 100), color='lightgreen', alpha=0.3)
                    elif conversion_rates[i] < conversion_rates[0]:
                        lower_critical_value = norm.ppf(sidak_alpha, loc=conversion_rates[i], scale=se_list[i])
                        plt.fill_between(x_range * 100, pdf, where=(x_range * 100 <= lower_critical_value * 100), color='lightcoral', alpha=0.3)

            plt.xlabel('Conversion rate (%)')
            plt.ylabel('Probability density')
            plt.title('Comparison of distributed conversion rates')
            plt.legend()
            st.pyplot(plt)
            plt.clf()

            # Output and conclusions
            st.write("### SRM Check")
            if srm_p_value > 0.01:
                st.write("This test is <span style='color: #009900; font-weight: 600;'>valid</span>. The distribution is as expected.", unsafe_allow_html=True)
            else:
                st.write("This test is <span style='color: #FF6600; font-weight: 600;'>invalid</span>: The distribution of traffic shows a statistically significant deviation from the expected values. Interpret the results with caution and check the distribution.", unsafe_allow_html=True)

            if num_variants >= 3:
                st.write("### Šidák Correction applied")
                st.write("The Šidák correction to solve for the Multiple Comparison Problem was applied due to 3 or more variants in the test.")

            def perform_superiority_test(i, alphabet, p_values, conversion_rates):
                if significant_results[i - 1]:
                    st.write(f"### Superiority test results for {alphabet[i]} vs {alphabet[0]}")
                    st.markdown(f" * Statistically significant result for {alphabet[i]} with p-value: {p_values[i-1]:.4f} and a power of {observed_powers[i-1] * 100:.2f}%!")
                    st.markdown(f" * Conversion rate change for {alphabet[i]}: {(conversion_rates[i] - conversion_rates[0]) * 100:.2f}%")
                    if conversion_rates[i] > conversion_rates[0]:
                        st.write(f"Variant {alphabet[i]} is a <span style='color: #009900; font-weight: 600;'>winner</span>, congratulations!", unsafe_allow_html=True)
                    else:
                        st.write(f"<span style='color: #FF6600; font-weight: 600;'>Loss prevented</span> with variant {alphabet[i]}! Congratulations with this valuable insight.", unsafe_allow_html=True)

            def perform_non_inferiority_test(i, alphabet, p_values, conversion_rates, visitor_counts, alpha_noninf, non_inferiority_margin):
                pooled_se = np.sqrt((conversion_rates[0] * (1 - conversion_rates[0]) / visitor_counts[0]) + (conversion_rates[i] * (1 - conversion_rates[i]) / visitor_counts[i]))
                z_stat_noninf = (conversion_rates[i] - conversion_rates[0] - non_inferiority_margin) / pooled_se
                p_value_noninf = stats.norm.cdf(z_stat_noninf)

                confidence_interval = stats.norm.interval(1 - alpha_noninf, loc=(conversion_rates[i] - conversion_rates[0]), scale=pooled_se)
                lower_bound, upper_bound = confidence_interval

                if p_values[i - 1] > sidak_alpha:
                    st.write(f"### Non-inferiority test results for {alphabet[i]} vs {alphabet[0]}:")
                    st.markdown(f" * Confidence interval for difference in conversion rates: ({lower_bound:.4f}, {upper_bound:.4f})")
                    st.markdown(f" * P-value (non-inferiority test): {p_value_noninf:.4f}")
                    if p_value_noninf <= alpha_noninf:
                        st.write(f"The test result for {alphabet[i]} vs {alphabet[0]} is not statistically significant in the Z-test with p-value {p_values[i-1]:.4f}, a power of {observed_powers[i-1] * 100:.2f}% \
                                 and a conversion rate change of {(conversion_rates[i] - conversion_rates[0]) * 100:.2f}%, \
                                 but this variant likely generates at least the same number of conversions as the control variant. \
                                 You may need to collect more data, or there is no real effect to be found.", unsafe_allow_html=True)
                    else:
                        st.write(f"The test result for {alphabet[i]} vs {alphabet[0]} is not statistically significant in the Z-test with p-value {p_values[i-1]:.4f}, a power of {observed_powers[i-1] * 100:.2f}% \
                                 and a conversion rate change of {(conversion_rates[i] - conversion_rates[0]) * 100:.2f}%, \
                                 but this variant likely won't generate at least the same number of conversions as the control variant.", unsafe_allow_html=True)

            # Main logic
            for i in range(1, num_variants):
                perform_superiority_test(i, alphabet, p_values, conversion_rates)

                alpha_noninf = 0.05
                non_inferiority_margin = 0.01
                perform_non_inferiority_test(i, alphabet, p_values, conversion_rates, visitor_counts, alpha_noninf, non_inferiority_margin)

        else:
            st.write("")
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()