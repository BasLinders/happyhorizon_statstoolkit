import streamlit as st
import scipy.stats as stats
import statistics
import string

st.set_page_config(
    page_title="SRM calculator",
    page_icon="🔢",
)

def run():
    st.title("Sample Ratio Mismatch (SRM) Checker")
    """
    This calculator lets you see if your online experiment correctly divided visitors among the variants, or if something went wrong and there was a mismatch with 
    the expected amount of visitors per variant. Enter the values below to get started. 

    Happy Learning!
    """

    num_variants = st.number_input("How many variants did your experiment have (including control)?", min_value=2, max_value=26, step=1)
    st.session_state.setdefault('visitor_counts', [0] * num_variants)
    st.session_state.setdefault('expected_proportions', [0.0] * num_variants)

    # Resize lists if `num_variants` has changed, while preserving existing values
    if num_variants != len(st.session_state.visitor_counts):
        st.session_state.visitor_counts = (st.session_state.visitor_counts[:num_variants] + [0] * num_variants)[:num_variants]
        st.session_state.expected_proportions = (st.session_state.expected_proportions[:num_variants] + [0.0] * num_variants)[:num_variants]

    # Display headers
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Visitors")
    with col2:
        st.write("### Expected Percentage")

    # Alphabet for variant labels (up to 26 variants)
    alphabet = string.ascii_uppercase

    # Generate input fields for each variant dynamically
    for i in range(num_variants):
        with col1:
            st.session_state.visitor_counts[i] = st.number_input(
                f"How many visitors did variant {alphabet[i]} have?",
                min_value=0, step=1, value=st.session_state.visitor_counts[i]
            )
        with col2:
            st.session_state.expected_proportions[i] = st.number_input(
                f"What percentage of users should be in variant {alphabet[i]}?",
                min_value=0.0, max_value=100.0, step=0.01, value=float(st.session_state.expected_proportions[i])
            )

    # Assign session state values to 'normal' variables for further processing
    visitor_counts = st.session_state.visitor_counts
    expected_proportions = st.session_state.expected_proportions

    st.write("")
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
                conclusion = "<span style='color: #FF6600; font-weight: 600;'>possible sample ratio mismatch</span>"
                srm_result = (
                    f"{conclusion}! The distribution of data between your variants significantly deviates from the "
                    f"expected proportions of {expected_distribution}. Check the distribution."
                )
            else:
                conclusion = "<span style='color: #009900; font-weight: 600;'>valid distribution</span>"
                srm_result = (
                    f"{conclusion}. The amount of visitors per variant does not significantly deviate from the expected split"
                    )

            # Display results
            st.write("")
            st.write("### Conclusion")
            st.write(f"P-value: {p_value:.4f}. The expected amount of visitors per variant on average is {round(statistics.mean(expected))}.") 
            st.write(f"This suggests a {srm_result}.", unsafe_allow_html=True)
            if num_variants > 2:
                st.write("### Expected vs. Observed Values")
                st.write("The specific distribution for your entered data:")
                data = {"Variant" : [alphabet[i] for i in range(num_variants)],
                        "Observed" : observed,
                        "Expected" : [round(x) for x in expected]}
                st.dataframe(data)

if __name__ == "__main__":
    run()