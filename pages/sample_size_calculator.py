import streamlit as st
from scipy.stats import norm
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Sample Size / MDE Calculator",
    page_icon="🔢",
)

# Input for MDE calculations
def get_mde_user_input(num_variants): 
    #st.session_state.num_variants = num_variants

    st.write("### Baseline Data for MDE calculations")
    st.write("Enter weekly visitors and weekly conversions.")
    st.session_state.baseline_visitors = st.number_input("Visitors in baseline variant:", min_value=0, step=1, value=st.session_state.baseline_visitors, key="mde_baseline_visitors") # Add key
    st.session_state.baseline_conversions = st.number_input("Conversions in baseline variant:", min_value=0, step=1, value=st.session_state.baseline_conversions, key="mde_baseline_conversions") # Add key

    # Additional inputs for confidence level, power, and tails
    st.session_state.risk = st.number_input("Desired confidence level (e.g., 90%):", min_value=0, max_value=100, step=1, value=st.session_state.risk, key="mde_risk") # Add key
    st.session_state.trust = st.number_input("Minimum trustworthiness (Power) (e.g., 80%):", min_value=0, max_value=100, step=1, value=st.session_state.trust, key="mde_trust") # Add key
    st.session_state.tails = st.selectbox(
        "Hypothesis type ('Greater' or 'Two-sided'): ",
        options=['Greater', 'Two-sided'], index=['Greater', 'Two-sided'].index(st.session_state.tails), key="mde_tails" # Add key
    )

    return st.session_state.num_variants, st.session_state.baseline_visitors, st.session_state.baseline_conversions, st.session_state.risk, st.session_state.trust, st.session_state.tails

# Input for sample size calculations
def get_sample_user_input(num_variants):
    #st.session_state.num_variants = num_variants

    st.write("### Baseline Data for Sample Size calculations")
    st.write("Enter weekly visitors and weekly conversions.")
    st.session_state.baseline_visitors = st.number_input("Visitors in baseline variant:", min_value=0, step=1, value=st.session_state.baseline_visitors, key="sample_baseline_visitors") # Add key
    st.session_state.baseline_conversions = st.number_input("Conversions in baseline variant:", min_value=0, step=1, value=st.session_state.baseline_conversions, key="sample_baseline_conversions") # Add key

    # Additional inputs for confidence level, power, and tails
    st.session_state.risk = st.number_input("Desired confidence level (e.g., 90%):", min_value=0, max_value=100, step=1, value=st.session_state.risk, key="sample_risk") # Add key
    st.session_state.trust = st.number_input("Minimum trustworthiness (Power) (e.g., 80%):", min_value=0, max_value=100, step=1, value=st.session_state.trust, key="sample_trust") # Add key
    st.session_state.tails = st.selectbox(
        "Hypothesis type ('Greater' or 'Two-sided'): ",
        options=['Greater', 'Two-sided'], index=['Greater', 'Two-sided'].index(st.session_state.tails), key="sample_tails" # Add key
    )
    st.session_state.mde = st.number_input("What MDE are you aiming for?", min_value=1, max_value=100, step=1, value=st.session_state.mde, key="sample_mde") # Add key

    return num_variants, st.session_state.baseline_visitors, st.session_state.baseline_conversions, st.session_state.risk, st.session_state.trust, st.session_state.tails, st.session_state.mde

# Holm-Bonferroni correction for MDE calculation
def holm_bonferroni_adjusted_z(num_variants, alpha, tails):
    adjusted_alpha = alpha / np.arange(num_variants, 0, -1)
    if tails == 'Two-sided':
        z_alpha = norm.ppf(1 - adjusted_alpha / 2)
    else:
        z_alpha = norm.ppf(1 - adjusted_alpha)
    return np.max(z_alpha)

def perform_mde_calculation(num_variants):
    num_variants, baseline_visitors, baseline_conversions, risk, trust, tails = get_mde_user_input()

    alpha = 1 - (risk / 100)
    power = trust / 100

    # Calculate baseline conversion rate
    baseline_rate = baseline_conversions / baseline_visitors

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

    return results

def display_mde_table(num_variants):
    num_variants, _, _, _, _, _ = get_mde_user_input(num_variants)
    results = perform_mde_calculation(num_variants)

    # Display results in a DataFrame
    df = pd.DataFrame(results, columns=['Week', 'Visitors / Variant', 'Relative MDE (%)'])
    df['Relative MDE (%)'] = df['Relative MDE (%)'].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

    # Display the DataFrame
    st.write("## MDE Calculation Results")
    st.write("""
        This table displays the minimum effect size detectable each week. An MDE of <5% is usually testworthy; 5-10% is debatable.
        For larger MDEs, consider whether the effect size can be achieved.
    """)
    if num_variants > 2:
        st.write("")
        st.write("Note: Because you entered more than 2 variants, the Holm-Bonferroni correction was applied.")
    st.write(df.to_html(index=False), unsafe_allow_html=True)

def calculate_sample_size(num_variants, baseline_visitors, baseline_conversions, mde, risk, trust, tails):

    # --- Input Validation and Parameter Conversion ---

    # Basic validation (although more comprehensive validation should ideally happen in `run`)
    if baseline_visitors <= 0:
        st.error("Baseline visitors must be greater than 0.")
        return # Stop calculation
    if baseline_conversions < 0: # Allow 0 conversions, but not negative
        st.error("Baseline conversions cannot be negative.")
        return
    if baseline_conversions > baseline_visitors:
        st.error("Baseline conversions cannot be greater than baseline visitors.")
        return
    if mde <= 0:
        st.error("Minimum Detectable Effect (MDE) must be greater than 0%.")
        return # Stop calculation

    try:
        mde_relative = mde / 100
        alpha = 1 - (risk / 100)  # Significance level (e.g., 0.05 for 95% confidence)
        power = trust / 100       # Statistical power (e.g., 0.8 for 80% power)
        beta = 1 - power          # Type II error rate

        # Baseline conversion rate
        p = baseline_conversions / baseline_visitors

        # Minimum Detectable Effect (absolute)
        mde_absolute = p * mde_relative
        effect_size = mde_absolute # Use a consistent name

        # Expected conversion rates
        p1 = p  # Control group rate
        p2 = p + mde_absolute  # Treatment group rate

        if p2 > 1.0:
            st.warning(f"The calculated treatment conversion rate ({p2:.2%}) exceeds 100% based on the baseline rate ({p:.2%}) and MDE ({mde}%). Please check your inputs.")
            # Allow calculation to proceed but warn user
        if p2 < 0.0:
             st.warning(f"The calculated treatment conversion rate ({p2:.2%}) is negative based on the baseline rate ({p:.2%}) and MDE ({mde}%). Please check your inputs.")
             # Allow calculation to proceed but warn user


    except ZeroDivisionError:
        st.error("Error during parameter calculation (potentially division by zero). Please ensure baseline visitors > 0.")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during parameter setup: {e}")
        return

    st.write("## Sample Size Calculation Results")
    st.write(f"Calculating required sample size for a desired relative MDE of **{mde}%**.")

    # --- Z-Score Calculation ---
    try:
        # Adjust alpha for multiple comparisons if necessary
        if num_variants > 2:
            # Assumes holm_bonferroni_adjusted_z returns the MAX Z-score needed
            num_comparisons = num_variants - 1
            z_alpha_adjusted = holm_bonferroni_adjusted_z(num_comparisons, alpha, tails)
            correction_applied = True
        else: # num_variants == 2
            if tails == 'Greater':
                z_alpha_adjusted = norm.ppf(1 - alpha)
            else: # Two-sided
                z_alpha_adjusted = norm.ppf(1 - alpha / 2)
            correction_applied = False

        # Z-score for power (positive value corresponding to 1-beta or power)
        z_power = norm.ppf(1 - beta) # Equivalent to norm.ppf(power)

        if z_alpha_adjusted is None or z_power is None:
             raise ValueError("Z-score calculation failed.")

    except AttributeError:
         st.error("Error: norm.ppf function not found. Ensure scipy is correctly installed.")
         return
    except Exception as e:
        st.error(f"An error occurred during Z-score calculation: {e}")
        return


    # --- Sample Size Formula ---
    try:
        # Variance terms
        var1 = p1 * (1 - p1)
        var2 = p2 * (1 - p2)

        # Ensure variances are non-negative (can happen with p=0 or p=1)
        var1 = max(var1, 0)
        var2 = max(var2, 0)

        # Use approximation p*(1-p) for the first term's variance (common)
        # Or use pooled variance: p_pooled = (p1+p2)/2; var_pooled = p_pooled*(1-p_pooled)
        term1 = z_alpha_adjusted * np.sqrt(2 * p * (1 - p)) # Using baseline variance approximation
        term2 = z_power * np.sqrt(var1 + var2)

        # Required sample size per group
        if effect_size == 0: # Should be caught by MDE > 0 validation earlier
             st.error("Effect size is zero. Cannot calculate sample size.")
             return

        sample_size_per_group = ((term1 + term2) ** 2) / (effect_size ** 2)
        sample_size_per_group = np.ceil(sample_size_per_group)

        if not np.isfinite(sample_size_per_group) or sample_size_per_group <= 0:
            st.error("Calculated sample size is invalid or non-positive. Please check inputs (especially MDE and baseline rates).")
            return

    except ZeroDivisionError:
        st.error("Error calculating sample size (division by zero). This might happen if the MDE is zero.")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during sample size calculation: {e}")
        return

    # --- Runtime Estimation ---
    def estimate_runtime(ss_per_group, visitors_per_week, n_variants):
        try:
            if visitors_per_week <= 0 or n_variants <= 0:
                return "infinite (zero baseline visitors or variants)"

            daily_visitors_total = visitors_per_week / 7
            visitors_per_group_per_day = daily_visitors_total / n_variants

            if visitors_per_group_per_day <= 0:
                return "infinite (zero daily visitors per group)"

            days_required = ss_per_group / visitors_per_group_per_day
            estimated_days = int(np.ceil(days_required))
            return estimated_days
        except Exception as e:
            st.warning(f"Could not estimate runtime: {e}")
            return "unavailable"

    estimated_days = estimate_runtime(sample_size_per_group, baseline_visitors, num_variants)

    # --- Display Results ---
    st.write(f"The required sample size per group (including control) is **{int(sample_size_per_group):,}**.")
    st.write(f"With an average of **{int(baseline_visitors):,}** total visitors per week, your test is estimated to run for approximately **{estimated_days}** days to reach the required sample size per group.")

    if correction_applied:
        st.write(f"*Note: The {type(holm_bonferroni_adjusted_z).__name__ if 'holm_bonferroni_adjusted_z' in globals() else 'configured multiple comparison correction'} was applied ({num_comparisons} comparisons) affecting the required significance level.*")
    elif num_variants == 2:
         st.write("*Note: Standard Z-scores used (no multiple comparison correction needed for 2 variants).*")
    
def run():
    # Initialize session state for inputs
    st.session_state.setdefault("num_variants", 2)
    st.session_state.setdefault("baseline_visitors", 0)
    st.session_state.setdefault("baseline_conversions", 0)
    st.session_state.setdefault("risk", 90)
    st.session_state.setdefault("tails", 'Greater')
    st.session_state.setdefault("trust", 80)
    st.session_state.setdefault("mde", 5)

    st.title("Sample Size Calculator")
    """
    This calculator provides a representative sample size and Minimum Detectable Effect (MDE) for your online experiment.
    The calculation will use baseline data and the number of variants to adjust the relative MDE, estimated over a 6-week period.
    
    Enter the values below to start.
    """

    st.write("")

    # Selectbox for choosing the calculation mode
    calculation_mode = st.selectbox(
        "Select Calculation Mode:",
        ("Calculate MDE based on Runtime", "Calculate Sample Size based on MDE")
    )

    num_variants = st.number_input("Number of variants (including control):", min_value=2, step=1, value=st.session_state.num_variants, key="num_variants")

    if calculation_mode == "Calculate MDE based on Runtime":
        get_mde_user_input(num_variants)  # Get inputs

        if st.button("Calculate MDE"):
            # Validate input
            if st.session_state.baseline_visitors <= 0 or st.session_state.baseline_conversions <= 0 or not (0 < st.session_state.risk <= 100) or not (0 < st.session_state.trust <= 100) or st.session_state.tails not in ['Greater', 'Two-sided']:
                st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)
            else:
                display_mde_table(num_variants)

    elif calculation_mode == "Calculate Sample Size based on MDE":
        get_sample_user_input(num_variants)  # Get inputs

        if st.button("Calculate Sample Size"):
            # Add Validation Block
            if (st.session_state.baseline_visitors <= 0 or
                    st.session_state.baseline_conversions < 0 or
                    st.session_state.baseline_conversions > st.session_state.baseline_visitors or
                    not (0 < st.session_state.risk <= 100) or
                    not (0 < st.session_state.trust <= 100) or
                    st.session_state.mde <= 0 or
                    st.session_state.tails not in ['Greater', 'Two-sided']):
                st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)
            else:
                calculate_sample_size(num_variants)


if __name__ == "__main__":
    run()