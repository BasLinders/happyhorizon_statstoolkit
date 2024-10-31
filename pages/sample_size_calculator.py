import streamlit as st
from scipy.stats import norm
import pandas as pd
import numpy as np

def run():
    st.set_page_config(
        page_title="Sample Size / MDE calculator",
        page_icon="🔢",
    )

    # Initialize session state for inputs
    st.session_state.setdefault("num_variants", 1)
    st.session_state.setdefault("baseline_visitors", [0] * st.session_state.num_variants)
    st.session_state.setdefault("baseline_conversions", [0] * st.session_state.num_variants)
    st.session_state.setdefault("risk", 90)
    st.session_state.setdefault("tails", 'Greater')
    st.session_state.setdefault("trust", 80)

    st.title("Sample Size Calculator")
    """
    This calculator provides you with a representative sample size and Minimum Detectable Effect (MDE) for your online experiment. 
    Enter the values below to start. The calculator dynamically adjusts for the number of variants in the experiment.
    Happy learning!
    """
    #num_variants = st.number_input("Number of variants (including control):", min_value=2, step=1, value=st.session_state.num_variants)

    # Resize lists if num_variants has changed, keeping previous values where possible
    if num_variants != st.session_state.num_variants:
        st.session_state.num_variants = num_variants
        st.session_state.baseline_visitors = (st.session_state.baseline_visitors[:num_variants] + [0] * num_variants)[:num_variants]
        st.session_state.baseline_conversions = (st.session_state.baseline_conversions[:num_variants] + [0] * num_variants)[:num_variants]

    # Gather baseline data per variant into temporary lists
    temp_visitors = st.session_state.baseline_visitors[:]
    temp_conversions = st.session_state.baseline_conversions[:]
    st.write("### Baseline Data per Variant")
    for i in range(st.session_state.num_variants):
        col1, col2 = st.columns(2)
        with col1:
            temp_visitors[i] = st.number_input(f"Visitors for Variant {i+1}", min_value=0, step=1, value=st.session_state.baseline_visitors[i])
        with col2:
            temp_conversions[i] = st.number_input(f"Conversions for Variant {i+1}", min_value=0, step=1, value=st.session_state.baseline_conversions[i])

    # Update session state lists after gathering inputs
    st.session_state.baseline_visitors = temp_visitors
    st.session_state.baseline_conversions = temp_conversions

    # Additional inputs
    st.session_state.risk = st.number_input("In %, enter the desired confidence level (e.g. 90):", min_value=0, max_value=100, step=1, value=st.session_state.risk)
    st.session_state.trust = st.number_input("In %, enter the minimum trustworthiness (Power) (e.g. 80):", min_value=0, max_value=100, step=1, value=st.session_state.trust)
    st.session_state.tails = st.selectbox("Do you want to know if B is better than A, or also the other way around ('Greater' or 'Two-sided)?", 
                                        options=['Greater', 'Two-sided'], index=['Greater', 'Two-sided'].index(st.session_state.tails))

    # Access variables for further use
    risk = st.session_state.risk
    trust = st.session_state.trust
    tails = st.session_state.tails
    baseline_visitors = st.session_state.baseline_visitors
    baseline_conversions = st.session_state.baseline_conversions

    st.write("")
    if st.button("Calculate Sample size and MDE"):
        if any(visitor <= 0 for visitor in baseline_visitors) or any(conversion <= 0 for conversion in baseline_conversions) or not (0 < risk <= 100) or not (0 < trust <= 100) or tails not in ['Greater', 'Two-sided']:
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)
        else:
            alpha = 1 - (risk / 100)
            power = trust / 100

            # Calculate baseline conversion rate for each variant
            baseline_rates = [conv / vis if vis > 0 else 0 for conv, vis in zip(baseline_conversions, baseline_visitors)]

            # Function for the Holm-Bonferroni correction
            def holm_bonferroni_adjusted_z(num_variants, alpha, tails=tails):
                adjusted_alpha = alpha / np.arange(num_variants, 0, -1)
                if tails == 'Two-sided':
                    z_alpha = norm.ppf(1 - adjusted_alpha / 2)
                else:
                    z_alpha = norm.ppf(1 - adjusted_alpha)
                return np.min(z_alpha)

            # Adjust alpha for multiple comparisons using the approximate Dunnett's adjustment
            if num_variants > 2:
                adjusted_z_alpha = holm_bonferroni_adjusted_z(num_variants - 1, alpha, tails)
            else:
                adjusted_z_alpha = alpha

            # Z-scores for power
            z_power = norm.ppf(power)

            # Weekly increments
            weeks = range(1, 7)  # For 6 weeks
            weekly_visitors_per_variant = np.ceil(sum(baseline_visitors) / num_variants)

            # Prepare a list to store the results for each week
            results = []
            for week in weeks:
                visitors_per_variant_weekly = int(weekly_visitors_per_variant * week)
                
                # Assuming constant conversion rate over weeks for simplicity; averaging baseline rates for simplicity
                avg_baseline_rate = np.mean(baseline_rates)

                # Standard error calculation for MDE
                se = np.sqrt(2 * avg_baseline_rate * (1 - avg_baseline_rate) / visitors_per_variant_weekly)

                # Absolute and relative MDE calculation
                mde_absolute = (adjusted_z_alpha + z_power) * se
                mde_relative = (mde_absolute / avg_baseline_rate) * 100  # Relative MDE as a percentage

                # Append results for this week to the list
                results.append([week, visitors_per_variant_weekly, mde_relative])

            # Convert the list of results into a DataFrame
            df = pd.DataFrame(results, columns=['Week', 'Visitors / Variant', 'Relative MDE'])
            df['Relative MDE'] = df['Relative MDE'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) and isinstance(x, (int, float)) else "N/A")
            df = df.reset_index(drop=True)

            # Display the DataFrame
            st.write("")
            st.write("### MDE table")
            st.write("""
                This table tells you what the minimum effect is that you need to see in order to reach statistical significance 
                for the number of weeks that your test has run. A relative MDE of < 5% is generally testworthy, 5-10% is debatable. For everything 
                above that, you should consider if the experiment will be able to achieve this effect and evaluate testworthiness.
                     
                * Please note: The Holm-Bonferroni correction is applied when entering > 2 variants.
            """)
            st.write(df.to_html(index=False), unsafe_allow_html=True)

if __name__ == "__main__":
    run()