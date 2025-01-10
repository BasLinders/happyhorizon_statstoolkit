import streamlit as st
from scipy.stats import norm
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Sample Size / MDE Calculator",
    page_icon="🔢",
)

def run():
    # Initialize session state for inputs
    st.session_state.setdefault("num_variants", 2)
    st.session_state.setdefault("baseline_visitors", 0)
    st.session_state.setdefault("baseline_conversions", 0)
    st.session_state.setdefault("risk", 90)
    st.session_state.setdefault("tails", 'Greater')
    st.session_state.setdefault("trust", 80)

    st.title("Sample Size Calculator")
    """
    This calculator provides a representative sample size and Minimum Detectable Effect (MDE) for your online experiment.
    The calculation will use baseline data and the number of variants to adjust the relative MDE, estimated over a 6-week period.
    
    Enter the values below to start.
    """
    
    # Input number of variants, but only one baseline for visitors and conversions
    num_variants = st.number_input("Number of variants (including control):", min_value=2, step=1, value=st.session_state.num_variants)
    st.session_state.num_variants = num_variants

    st.write("### Baseline Data")
    st.session_state.baseline_visitors = st.number_input("Visitors in baseline variant:", min_value=0, step=1, value=st.session_state.baseline_visitors)
    st.session_state.baseline_conversions = st.number_input("Conversions in baseline variant:", min_value=0, step=1, value=st.session_state.baseline_conversions)

    # Additional inputs for confidence level, power, and tails
    st.session_state.risk = st.number_input("Desired confidence level (e.g., 90%):", min_value=0, max_value=100, step=1, value=st.session_state.risk)
    st.session_state.trust = st.number_input("Minimum trustworthiness (Power) (e.g., 80%):", min_value=0, max_value=100, step=1, value=st.session_state.trust)
    st.session_state.tails = st.selectbox(
        "Hypothesis type ('Greater' or 'Two-sided'): ", 
        options=['Greater', 'Two-sided'], index=['Greater', 'Two-sided'].index(st.session_state.tails)
    )

    # Access variables for calculations
    risk = st.session_state.risk
    trust = st.session_state.trust
    tails = st.session_state.tails
    baseline_visitors = st.session_state.baseline_visitors
    baseline_conversions = st.session_state.baseline_conversions

    st.write("")
    if st.button("Calculate Sample Size and MDE"):
        if baseline_visitors <= 0 or baseline_conversions <= 0 or not (0 < risk <= 100) or not (0 < trust <= 100) or tails not in ['Greater', 'Two-sided']:
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)
        else:
            alpha = 1 - (risk / 100)
            power = trust / 100

            # Calculate baseline conversion rate
            baseline_rate = baseline_conversions / baseline_visitors

            # Holm-Bonferroni correction for MDE calculation
            def holm_bonferroni_adjusted_z(num_variants, alpha, tails=tails):
                adjusted_alpha = alpha / np.arange(num_variants, 0, -1)
                if tails == 'Two-sided':
                    z_alpha = norm.ppf(1 - adjusted_alpha / 2)
                else:
                    z_alpha = norm.ppf(1 - adjusted_alpha)
                return np.min(z_alpha)

            # Adjust alpha for multiple comparisons
            if num_variants > 2:
                adjusted_z_alpha = holm_bonferroni_adjusted_z(num_variants - 1, alpha, tails)
            else:
                adjusted_z_alpha = norm.ppf(1 - alpha) if tails == 'Greater' else norm.ppf(1 - alpha / 2)

            # Z-score for power
            z_power = norm.ppf(power)

            # Weekly increments
            weeks = range(1, 7)  # For 6 weeks
            weekly_visitors = int(np.ceil(baseline_visitors / num_variants))

            # Prepare list to store results
            results = []
            for week in weeks:
                visitors_per_variant_weekly = weekly_visitors * week

                # Standard error and MDE calculations
                se = np.sqrt(2 * baseline_rate * (1 - baseline_rate) / visitors_per_variant_weekly)
                mde_absolute = (adjusted_z_alpha + z_power) * se
                mde_relative = (mde_absolute / baseline_rate) * 100

                # Store results
                results.append([week, visitors_per_variant_weekly, mde_relative])

            # Display results in a DataFrame
            df = pd.DataFrame(results, columns=['Week', 'Visitors / Variant', 'Relative MDE (%)'])
            df['Relative MDE (%)'] = df['Relative MDE (%)'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

            # Display the DataFrame
            st.write("### MDE Table")
            st.write("""
                This table displays the minimum effect size detectable each week. An MDE of <5% is usually testworthy; 5-10% is debatable.
                For larger MDEs, consider whether the effect size can be achieved.
            """)
            if num_variants > 2:
                st.write("")
                st.write("Note: Because you entered more than 2 variants, the Holm-Bonferroni correction was applied.")
            st.write(df.to_html(index=False), unsafe_allow_html=True)

if __name__ == "__main__":
    run()
