try:
    import scipy.stats as stats
    import streamlit as st
    print("All libraries have been successfully loaded.")
except ImportError as e:
    print(f"These libraries failed to load: {e}")

# Sample data
while True:
    try:
        visitors_a = st.number_input("How many visitors did A have? ")
        visitors_b = st.number_input("How many visitors did B have? ")
        if visitors_a < 0 or visitors_b < 0:
            raise ValueError("Visitor counts cannot be negative.")
        break
    except ValueError as e:
        print(f"Error: {e}. Please enter valid visitor counts.")

proportion_a = st.number_input("What percentage of users should be in A? ")
proportion_b = st.number_input("What percentage of users should be in B? ")

observed = [visitors_a, visitors_b]

if proportion_a + proportion_b != 100:
    print("The total sample should be equal to 100.")
else:
    expected_distribution = [proportion_a / 100, proportion_b / 100]

    # Calculate expected frequencies based on observed data and expected distribution
    total_visitors = sum(observed)
    expected = [total_visitors * p for p in expected_distribution]

    # Perform the chi-squared test
    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    # Define SRM result based on p-value threshold
    if p_value < 0.01:
        srm_result = (
            f"possible sample ratio mismatch! The distribution of data significantly deviates from the "
            f"expected proportions of {expected_distribution}. Check the distribution."
        )
    else:
        srm_result = "valid distribution. The sample proportions do not significantly deviate from the expected split."

    # Print results
    print(f"p-value: {p_value:.4f}. This suggests a {srm_result}.")
