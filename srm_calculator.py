import streamlit as st
import scipy.stats as stats
import string

st.title("Sample Ratio Mismatch (SRM) Checker")

# User input for number of variants
num_variants = st.number_input("How many variants did your experiment have?", min_value=2, max_value=26, step=1)

# Dynamically generate input fields for visitor counts and expected proportions
visitor_counts = []
expected_proportions = []
alphabet = string.ascii_uppercase

for i in range(num_variants):
    visitor_counts.append(st.number_input(f"How many visitors did variant {alphabet[i]} have?", min_value=0, step=1))
    expected_proportions.append(st.number_input(f"What percentage of users should be in variant {alphabet[i]}?", min_value=0.0, max_value=100.0, step=0.01))

if st.button("Check for Sample Ratio Mismatch"):
    if sum(expected_proportions) != 100:
        st.error("The total sample proportion should be equal to 100.")
    else:
        observed = visitor_counts
        expected_distribution = [p / 100 for p in expected_proportions]

        # Calculate expected frequencies based on observed data and expected distribution
        total_visitors = sum(observed)
        expected = [total_visitors * p for p in expected_distribution]

        # Perform the chi-squared test
        chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

        # Define SRM result based on p-value threshold
        if p_value < 0.01:
            srm_result = (
                f"possible sample ratio mismatch! The distribution of data between your variants significantly deviates from the "
                f"expected proportions of {expected_distribution}. Check the distribution"
            )
        else:
            srm_result = "valid distribution. The amount of visitors per variant does not significantly deviate from the expected split"

        # Display results
        st.write(f"p-value: {p_value:.4f}. \nThis suggests a {srm_result}.")
