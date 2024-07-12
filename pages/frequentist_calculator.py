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

    st.write("# Frequentist Calculator")
    st.markdown("""
    This calculator tests your data against the null-hypothesis (= there's no difference between 'A' and 'B'). If your results deviate significantly from the null-
    hypothesis, that means that the change you tested did indeed shift user behavior for your chosen metric. 

    If you enter less than 1000 users, power calculations (= trustworthiness of the measured effect) are done with a bootstrapping method, 
    meaning your PC will run 10,000 simulations to find the precise power for your test data. 
    If you enter over 1000 users per variant, the calculator will use an analytical approach to get an estimate.

    The calculator will output the results at the very bottom, but will also display relevant statistics for your data.

    Enter your experiment values below. Happy learning!
    """)

    num_variants = st.number_input("How many variants did your experiment have (including control)?", min_value=2, max_value=10, step=1)
    col1, col2 = st.columns(2)

    # Generate fields for variants
    visitor_counts = []
    variant_conversions = []
    alphabet = string.ascii_uppercase

    with col1:
        st.write("### Visitors")
    with col2:
        st.write("### Conversions")

    # Experiment inputs
    for i in range(num_variants):
        with col1:
            visitor_counts.append(st.number_input(f"How many visitors did variant {alphabet[i]} have?", min_value=0, step=1))
        with col2:
            variant_conversions.append(st.number_input(f"How many conversions did variant {alphabet[i]} have?", min_value=0, step=1))

    risk = st.number_input("How much risk do you want to take in % (enter 5, 10, etc)?", min_value=1, step=1)
    tail = st.selectbox("Do you only want to know if B is better than A ('greater'), or if B is worse than A ('two-sided')?", options=['greater', 'two-sided'])

    # Ensure inputs are valid before performing calculations
    valid_inputs = all(v > 0 for v in visitor_counts) and all(0 <= c <= v for c, v in zip(variant_conversions, visitor_counts))

    if valid_inputs:
        alpha = risk / 100

        # Verify the data
        st.write("### Please verify your input:")
        st.write(f"Chosen threshold for significance: {round((1 - alpha) * 100)}%")
        st.write(f"Chosen test type: {'B is better than A' if tail == 'greater' else 'B is worse than A'}.")

        for i in range(num_variants):
            st.write(f"Variant {alphabet[i]}: {visitor_counts[i]} visitors, {variant_conversions[i]} conversions")

        # Conversion rates
        conversion_rates = [c / v for c, v in zip(variant_conversions, visitor_counts)]
        for i in range(num_variants):
            st.write(f"Conversion Rate {alphabet[i]}: {conversion_rates[i] * 100:.2f}%")

        # SRM check
        observed = visitor_counts
        expected = [sum(observed) / len(observed)] * len(observed)

        # Perform the chi-squared test for independence
        contingency_table = np.array([observed, expected])
        chi2, srm_p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)
        st.write(f"SRM p-value: {srm_p_value:.4f}")

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
            st.write(f"Z-statistic for {alphabet[i]} vs {alphabet[0]}: {z_stats[i-1]:.4f}")
            st.write(f"P-value for {alphabet[i]} vs {alphabet[0]}: {p_values[i-1]:.4f}")

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
                st.write(f"Observed power for {alphabet[i]} vs {alphabet[0]}: {observed_powers[i-1] * 100:.2f}%")

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
                st.write(f"Observed power for {alphabet[i]} vs {alphabet[0]}: {observed_powers[i-1] * 100:.2f}%")

        st.write("")
        st.write("### Probability Density Graphs:")

        # Probability density graph
        plt.figure(figsize=(10, 6))

        # Define x_range dynamically based on the conversion rates
        min_conversion_rate = min(conversion_rates)
        max_conversion_rate = max(conversion_rates)
        x_min = min_conversion_rate - 0.01  # Extend the range slightly to the left
        x_max = max_conversion_rate + 0.01  # Extend the range slightly to the right
        x_range = np.linspace(x_min, x_max, 1000)

        colors = ['#FF5733', '#3498DB', '#9B59B6', '#E74C3C', '#1ABC9C', '#F39C12', '#2980B9', '#D35400', '#C0392B', '#7D3C98']

        for i in range(num_variants):
            pdf = norm.pdf(x_range, conversion_rates[i], se_list[i])
            plt.plot(x_range * 100, pdf, label=f'Variant {alphabet[i]}', color=colors[i])
            plt.axvline(conversion_rates[i] * 100, color=colors[i], linestyle='--')
            plt.text(conversion_rates[i] * 100, plt.ylim()[1] * 0.50, f'Mean {alphabet[i]}', color=colors[i], ha='right', rotation=90, va='bottom')

            # Shading the regions for winners and losers
            if p_values[i-1] <= alpha:
                if conversion_rates[i] > conversion_rates[0]:
                    # Shade only the upper tail in light green
                    upper_critical_value = norm.ppf(1 - alpha, loc=conversion_rates[i], scale=se_list[i])
                    plt.fill_between(x_range * 100, pdf, where=(x_range >= upper_critical_value), color='lightgreen', alpha=0.3)
                elif conversion_rates[i] < conversion_rates[0]:
                    # Shade only the lower tail in light red
                    lower_critical_value = norm.ppf(alpha, loc=conversion_rates[i], scale=se_list[i])
                    plt.fill_between(x_range * 100, pdf, where=(x_range <= lower_critical_value), color='lightcoral', alpha=0.3)

        # Adjust x-axis for percentages and add legend, titles, and labels
        plt.xlabel('Conversion rate (%)')
        plt.ylabel('Probability')
        plt.title('Comparison of distributed conversion rates')
        plt.legend()
        st.pyplot(plt)
        plt.clf()

        # Output and conclusions
        st.write("")
        st.write("### Test results:")
        if srm_p_value > 0.01:
            st.write("This test is valid. The distribution is as expected.")
        else:
            st.write("This test is invalid: The distribution of traffic shows a statistically significant deviation from the expected values. Interpret the results with caution and check the distribution.")

        for i in range(1, num_variants):
            if p_values[i-1] <= alpha:
                st.write(f"Statistically significant result for {alphabet[i]} with p-value: {p_values[i-1]:.4f}!")
                st.write(f"Conversion rate change for {alphabet[i]}: {(conversion_rates[i] - conversion_rates[0]) * 100:.2f}%")
                if conversion_rates[i] > conversion_rates[0]:
                    st.write(f"Variant {alphabet[i]} is a winner, congratulations!")
                else:
                    st.write(f"Loss prevented with variant {alphabet[i]}! Congratulations with this valuable insight.")
            else:
                st.write(f"No significant result in the Z-test for {alphabet[i]} with p-value: {p_values[i-1]:.4f}.")
                st.write("")
                # Non-inferiority test
                alpha_noninf = 0.05  # Minimize the chance of disqualifying B to 5%
                non_inferiority_margin = 0.01  # Maximum acceptable difference in effect size for B to still be considered "non-inferior."

                st.write("### Non-inferiority Test Results:")
                for i in range(1, num_variants):
                    pooled_se = np.sqrt((conversion_rates[0] * (1 - conversion_rates[0]) / visitor_counts[0]) + 
                                        (conversion_rates[i] * (1 - conversion_rates[i]) / visitor_counts[i]))
                    z_stat_noninf = (conversion_rates[i] - conversion_rates[0] - non_inferiority_margin) / pooled_se
                    p_value_noninf = stats.norm.cdf(z_stat_noninf)
                    
                    confidence_interval = stats.norm.interval(1 - alpha_noninf, loc=(conversion_rates[i] - conversion_rates[0]), scale=pooled_se)
                    lower_bound, upper_bound = confidence_interval
                    st.write(f"Variant {alphabet[i]} vs {alphabet[0]}:")
                    st.write(f"Confidence interval for difference in conversion rates: ({lower_bound:.4f}, {upper_bound:.4f})")
                    st.write(f"P-value (non-inferiority test): {p_value_noninf:.4f}")

                    if p_value_noninf <= alpha_noninf:
                        st.write("The test result is not statistically significant, but variant B generates at least the same number of conversions as variant A.")
                    else:
                        st.write("The test result is not statistically significant and variant B possibly will not generate at least the same number of conversions as variant A.")

    else:
        st.write("")
        st.write("Please enter valid inputs for all fields.")

if __name__ == "__main__":
    run()