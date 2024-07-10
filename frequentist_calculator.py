import streamlit as st
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import concurrent.futures

st.write("# Frequentist Calculator")
linkedin_url = "https://www.linkedin.com/in/blinders/"
happyhorizon_url = "https://happyhorizon.com/"
footnote_text = f"""Designed and developed by <a href="{linkedin_url}" target="_blank">Bas Linders</a> @<a href="{happyhorizon_url}" target="_blank">Happy Horizon.</a>"""
st.markdown(footnote_text, unsafe_allow_html=True)
st.write("")
"""
This calculator tests your data against the null-hypothesis (= there's no difference between 'A' and 'B'). If your results deviate significantly from the null-
hypothesis, that means that the change you tested did indeed shift user behavior for your chosen metric. 

If you enter less than 1000 users, power calculations (= trustworthiness of the measured effect) are done with a bootstrapping method, 
meaning your PC will run 10,000 simulations to find the precise power for your test data. 
If you enter over 1000 users per variant, the calculator will use an analytical approach to get an estimate.

The calculator will output the results at the very bottom, but will also display relevant statistics for your data.

Enter your experiment values below. Happy learning!
"""

# Experiment inputs
visitors_a = st.number_input("How many visitors does 'A' have?", min_value=0, step=1)
visitors_b = st.number_input("How many visitors does 'B' have?", min_value=0, step=1)
conversions_a = st.number_input("How many conversions does 'A' have?", min_value=0, step=1)
conversions_b = st.number_input("How many conversions does 'B' have?", min_value=0, step=1)
risk = st.number_input("How much risk do you want to take in % (enter 5, 10, etc)?", min_value=1, step=1)
tail = st.selectbox("Do you only want to know if B is better than A ('greater'), or if B is worse than A ('two-sided')?", options=['greater', 'two-sided'])

# Ensure inputs are valid before performing calculations
if visitors_a > 0 and visitors_b > 0 and conversions_a > 0 and conversions_b > 0 and conversions_a <= visitors_a and conversions_b <= visitors_b:
    alpha = risk / 100
    uplift = (conversions_b - conversions_a) / conversions_a if conversions_a != 0 else 0

    # Verify the data
    st.write("### Please verify your input:")
    st.write(f"Chosen threshold for significance: {round((1 - alpha) * 100)}%")
    st.write(f"Chosen test type: {'B is better than A' if tail == 'greater' else 'B is worse than A'}.")
    st.write(f"Variant A: {visitors_a} visitors, {conversions_a} conversions")
    st.write(f"Variant B: {visitors_b} visitors, {conversions_b} conversions")
    st.write(f"Measured change in conversion rate: {uplift * 100:.2f}%")

    def validate_inputs(visitors, conversions):
        if visitors == 0:
            raise ValueError("Visitors cannot be zero")
        if conversions > visitors:
            raise ValueError("Conversions cannot exceed the number of visitors")
        if conversions < 0 or visitors < 0:
            raise ValueError("Visitors and conversions cannot be negative")

    try:
        validate_inputs(visitors_a, conversions_a)
        validate_inputs(visitors_b, conversions_b)
    except ValueError as e:
        st.write(f"Input error: {e}")

    # Calculations

    # Conversion rates
    CR_A = conversions_a / visitors_a
    CR_B = conversions_b / visitors_b
    relative_change = (CR_B - CR_A) / CR_A
    st.write(f"Conversion Rate A: {CR_A * 100:.2f}%, Conversion Rate B: {CR_B * 100:.2f}%")
    st.write(f"Relative change: {relative_change * 100:.2f}%")

    # SRM check
    observed = [visitors_b, visitors_a]
    expected = [sum(observed) / 2, sum(observed) / 2]

    # Perform the chi-squared test
    chi2, srm_p_value, dof, ex = stats.chi2_contingency([observed, expected])
    #st.write(f"SRM p-value: {srm_p_value:.4f}")

    # Calculate sample proportions
    p_A = conversions_a / visitors_a
    p_B = conversions_b / visitors_b

    # Calculate the pooled proportion
    p_pooled = (conversions_a + conversions_b) / (visitors_a + visitors_b)

    # Calculate the standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / visitors_a + 1 / visitors_b))

    # Calculate the z-statistic
    z_stat = (p_B - p_A) / se

    # Calculate the p-value for a two-sided test
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Adjust for one-sided test if needed
    if tail == 'greater':
        p_value /= 2
        p_value = 1 - p_value if z_stat < 0 else p_value
    elif tail == 'less':
        p_value /= 2
        p_value = 1 - p_value if z_stat > 0 else p_value
    
    st.write("")
    st.write("### Test statistics:")
    #st.write(f"Z-statistic: {z_stat:.4f}")
    #st.write(f"P-value: {p_value:.4f}")

    # Standard Errors
    SE_A = np.sqrt(CR_A * (1 - CR_A) / visitors_a)
    SE_B = np.sqrt(CR_B * (1 - CR_B) / visitors_b)
    st.write(f"Standard error of A: {SE_A:.6f}, Standard error of B: {SE_B:.6f}")

    # Confidence intervals for conversion rates
    CI_A_lower = CR_A - 1.96 * SE_A
    CI_A_upper = CR_A + 1.96 * SE_A
    CI_B_lower = CR_B - 1.96 * SE_B
    CI_B_upper = CR_B + 1.96 * SE_B

    st.write(f"Confidence intervals A: [{CI_A_lower:.4f}, {CI_A_upper:.4f}]")
    st.write(f"Confidence intervals B: [{CI_B_lower:.4f}, {CI_B_upper:.4f}]")

    # Standard Error of the difference
    SE_diff = np.sqrt(SE_A**2 + SE_B**2)
    st.write(f"Standard error of difference: {SE_diff:.6f}")

    # Effect size (z-score)
    effect_size = (CR_B - CR_A) / SE_diff
    st.write(f"Effect size (z-score): {effect_size:.4f}")

    # Analytical power calculation function
    def analytical_power(CR_A, CR_B, visitors_a, visitors_b, alpha, tail):
        pooled_p = (conversions_a + conversions_b) / (visitors_a + visitors_b)
        effect_size = abs(CR_A - CR_B) / np.sqrt(pooled_p * (1 - pooled_p) * (1 / visitors_a + 1 / visitors_b))
        if tail in ['greater', 'less']:
            z_alpha = stats.norm.ppf(1 - alpha)
            power = stats.norm.cdf(effect_size - z_alpha) if tail == 'greater' else stats.norm.cdf(effect_size + z_alpha)
        else:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            power = stats.norm.cdf(effect_size - z_alpha) + stats.norm.cdf(-effect_size - z_alpha)
        return power

    # Power calculations
    if visitors_a > 1000 or visitors_b > 1000:
        observed_power = analytical_power(CR_A, CR_B, visitors_a, visitors_b, alpha, tail)
        st.write("")
        st.write("\nUsing analytical approach to calculate observed power...\n")
    else:
        st.write("")
        st.write("\nUsing bootstrapping to calculate power for more accuracy. Just a moment (running simulations)...\n")
        n_bootstraps = 10000

        def bootstrap_sample(data_a, data_b, alpha, tail):
            sample_A = np.random.choice(data_a, size=len(data_a), replace=True)
            sample_B = np.random.choice(data_b, size=len(data_b), replace=True)
            z_stat = (np.mean(sample_B) - np.mean(sample_A)) / np.sqrt(np.var(sample_A) / len(sample_A) + np.var(sample_B) / len(sample_B))
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            if tail == 'greater':
                p_value /= 2
                p_value = 1 - p_value if z_stat < 0 else p_value
            elif tail == 'less':
                p_value /= 2
                p_value = 1 - p_value if z_stat > 0 else p_value
            return p_value < alpha

        def bootstrap_power(data_a, data_b, alpha, tail, n_bootstraps=10000):
            significant_results = 0
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(bootstrap_sample, data_a, data_b, alpha, tail) for _ in range(n_bootstraps)]
                for future in concurrent.futures.as_completed(futures):
                    if future.result():
                        significant_results += 1
            return significant_results / n_bootstraps

        data_a = np.concatenate([np.ones(conversions_a), np.zeros(visitors_a - conversions_a)])
        data_b = np.concatenate([np.ones(conversions_b), np.zeros(visitors_b - conversions_b)])
        observed_power = bootstrap_power(data_a, data_b, alpha, tail, n_bootstraps=n_bootstraps)

    #st.write(f"Power test type: {tail}")
    st.write(f"Observed power: {observed_power * 100:.2f}%")
    st.write(f"Chosen risk level: {risk}%")
    #st.write(f"P-value: {p_value:.4f}")

    # Non-inferiority test
    alpha_noninf = 0.05  # minimize the chance of disqualifying B to 5%
    non_inferiority_margin = 0.01  # maximum acceptable difference in effect size for B to still be considered "non-inferior."

    pooled_se = np.sqrt((CR_A * (1 - CR_A) / visitors_b) + (CR_B * (1 - CR_B) / visitors_b))
    z_stat_noninf = (CR_B - CR_A - non_inferiority_margin) / pooled_se
    p_value_noninf = stats.norm.cdf(z_stat_noninf)

    # Confidence Interval
    confidence_interval = stats.norm.interval(1 - alpha_noninf, loc=(CR_B - CR_A), scale=SE_diff)
    lower_bound, upper_bound = confidence_interval
    print(f"Confidence interval for difference in conversion rates: ({lower_bound:.4f}, {upper_bound:.4f})")

    # Probability density graph

    alpha_graph = risk / 100  # Adjust risk to alpha level

    # Given range adjustments and PDF calculations
    multiplier = 3
    min_x = min(CR_A, CR_B) - multiplier * max(SE_A, SE_B)
    max_x = max(CR_A, CR_B) + multiplier * max(SE_A, SE_B)
    x_range = np.linspace(min_x, max_x, 1000)
    pdf_a = norm.pdf(x_range, CR_A, SE_A)
    pdf_b = norm.pdf(x_range, CR_B, SE_B)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_range * 100, pdf_a, label='Variant A')  # Convert to percentages
    plt.plot(x_range * 100, pdf_b, label='Variant B')  # Convert to percentages

    # Vertical lines
    plt.axvline(CR_A * 100, color='#FF5733', linestyle='--')
    plt.axvline(CR_B * 100, color='#3498DB', linestyle='--')

    # Text annotations for the vertical lines
    plt.text(CR_A * 100, plt.ylim()[1] * 0.50, 'Mean A', color='#FF5733', ha='right', rotation=90, va='bottom')
    plt.text(CR_B * 100, plt.ylim()[1] * 0.50, 'Mean B', color='#3498DB', ha='right', rotation=90, va='bottom')


    if p_value <= alpha_graph:
        if CR_B > CR_A:
            # If B is the winning variant, shade the upper tail of B
            upper_critical_value_b = norm.ppf(1-alpha_graph, CR_B, SE_B)
            plt.fill_between(x_range * 100, pdf_b, where=(x_range * 100 >= upper_critical_value_b * 100),
                             color='#90EE90', alpha=0.3)
        else:
            # If A is significantly worse, shade the upper tail of A
            upper_critical_value_a = norm.ppf(1-alpha_graph, CR_A)
            plt.fill_between(x_range * 100, pdf_a, where=(x_range * 100 >= upper_critical_value_a * 100),
                             color='#FFB6C1', alpha=0.3)

    # Adjust x-axis for percentages and add legend, titles, and labels
    plt.xlabel('Conversion rate (%)')
    plt.ylabel('Probability')
    plt.title('Comparison of distributed conversion rates')
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    # Output and conclusions
    # Overall decision statistics
    if srm_p_value > 0.01:
        st.write("")
        st.write("### Test results:")
        st.write("This test is valid. The distribution is as expected.")
    else:
        st.write("")
        st.write("Test results:")
        st.write("This test is invalid: The distribution of traffic shows a statistically significant deviation "
                 "from the expected values. Interpret the results with caution and check the distribution.")
    st.write(f"Observed power of this test: {observed_power * 100:.2f}%")
    #st.write(f"P-value (z-test): {p_value:.4f}\n")

    if p_value <= alpha:
        print(f"Confidence interval for difference in conversion rates: ({lower_bound:.4f}, {upper_bound:.4f})")
        st.write(f"Statistically significant test result with p-value: {p_value:.4f}!")
        st.write(f"Conversion rate change: {relative_change * 100:.2f}%")
        if relative_change > 0:
            st.write("Your variant is a winner, congratulations!")
        else:
            st.write("Loss prevented! Congratulations with this valuable insight.")
    elif p_value > alpha and p_value_noninf <= alpha_noninf:
        st.write(f"Conversion rate of A: {CR_A * 100:.3f}%")
        st.write(f"Conversion rate of B: {CR_B * 100:.3f}%")
        st.write(f"P-value (non-inferiority test): {p_value_noninf:.4f}")
        st.write("The test result is not statistically significant, but variant B generates at least "
                 "the same number of conversions as variant A.")
    else:
        st.write(f"Conversion rate of A: {CR_A * 100:.3f}%")
        st.write(f"Conversion rate of B: {CR_B * 100:.3f}%")
        st.write(f"P-value (non-inferiority test): {p_value_noninf:.4f}")
        st.write("The test result is not statistically significant and variant B possibly "
                 "will not generate at least the same number of conversions as variant A.")

else:
    st.write("")
    st.write("Please enter valid inputs for all fields.")