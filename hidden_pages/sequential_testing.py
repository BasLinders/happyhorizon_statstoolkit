import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests  # For error spending

## -- HELPER FUNCTIONS --
def log_likelihood_ratio(p0, p1, x, n):
    """Calculates the log-likelihood ratio for a given number of conversions."""
    eps = 1e-10
    p0 = np.clip(p0, eps, 1 - eps)
    p1 = np.clip(p1, eps, 1 - eps)
    loglr = x * np.log(p1 / p0) + (n - x) * np.log((1 - p1) / (1 - p0))
    return loglr

# Sequential Probability Ratio Test function (with error spending)
def sprt_test_error_spending(p0, p1, conversions_b_cumulative, visitors_b_cumulative,
                              alpha=0.05, beta=0.2, intervals=None, error_spending_function=None):
    """
    Performs a Sequential Probability Ratio Test (SPRT) with error spending.

    Args:
        p0 (float): Conversion rate under the null hypothesis (based on group A).
        p1 (float): Conversion rate under the alternative hypothesis (based on group B).
        conversions_b_cumulative (list): Cumulative number of conversions in group B at each interval.
        visitors_b_cumulative (list): Cumulative number of visitors in group B at each interval.
        alpha (float, optional): Overall Type I error rate. Defaults to 0.05.
        beta (float, optional): Overall Type II error rate. Defaults to 0.2.
        intervals (list, optional): List of indices (corresponding to the cumulative data)
                                     at which to check for early stopping. If None, checks only at the end.
        error_spending_function (callable, optional): A function that takes the overall alpha
                                                       and the proportion of information time and returns
                                                       the cumulative alpha spent. If None, no error spending is used.

    Returns:
        tuple: (llr_values, A_values, B_values, results) where:
            llr_values (list): List of log-likelihood ratios at each check interval.
            A_values (list): List of upper boundaries at each check interval.
            B_values (list): List of lower boundaries at each check interval.
            results (list): List of decisions at each check interval ("Reject H0", "Fail to reject H0", "Continue").
    """
    if intervals is None:
        intervals = [len(visitors_b_cumulative) - 1]  # Check only at the end

    llr_values = []
    A_values = []
    B_values = []
    results = []
    cumulative_alpha_spent = 0.0

    # Calculate boundaries based on overall alpha and beta initially
    A_overall = np.log((1 - beta) / alpha)
    B_overall = np.log(beta / (1 - alpha))

    for i in intervals:
        conversions_b = conversions_b_cumulative[i]
        visitors_b = visitors_b_cumulative[i]

        llr = log_likelihood_ratio(p0, p1, conversions_b, visitors_b)
        llr_values.append(llr)

        if error_spending_function:
            proportion_information = (visitors_b / visitors_b_cumulative[-1]) if visitors_b_cumulative[-1] > 0 else 0.0
            alpha_to_spend = error_spending_function(alpha, proportion_information) - cumulative_alpha_spent
            cumulative_alpha_spent += alpha_to_spend
            A = np.log((1 - beta) / alpha_to_spend) if alpha_to_spend > 0 else np.inf
            B = np.log(beta / (1 - (cumulative_alpha_spent + (alpha - cumulative_alpha_spent)))) if (1 - (cumulative_alpha_spent + (alpha - cumulative_alpha_spent))) > 0 else -np.inf
        else:
            A = A_overall
            B = B_overall

        A_values.append(A)
        B_values.append(B)

        if llr >= A:
            results.append("Reject H0: Significant difference in favor of B.")
            break  # Stop early if H0 is rejected
        elif llr <= B:
            results.append("Fail to reject H0: No significant difference detected (or evidence favors A).")
            break  # Stop early if H0 is not rejected
        else:
            results.append("Continue testing: No conclusive result yet.")

    return llr_values, A_values, B_values, results

st.write("SPRT function with error spending defined.")

# Example error spending functions

def o_brien_fleming_spending(alpha, proportion):
    """O'Brien-Fleming type error spending function."""
    if proportion == 0:
        return 0
    return 2 * alpha * (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(proportion)))

def pocock_spending(alpha, proportion):
    """Pocock type error spending function (constant spending rate)."""
    # return alpha * np.log(1 + (np.exp(1) - 1) * proportion)
    return 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(proportion)))

def haybittle_peto_spending(alpha, proportion):
    """Haybittle-Peto type error spending function (conservative early spending)."""
    if proportion < 0.8:
        return 0.01 * alpha
    else:
        return alpha

# --- Assuming variables are defined before this point ---
# Example data for cumulative conversions and visitors at different intervals
cumulative_conversions_b = [10, 25, 45, 70]
cumulative_visitors_b = [100, 250, 400, 650]

# Define the intervals at which you want to check (e.g., after each data point)
check_intervals = list(range(len(cumulative_visitors_b)))

# Run SPRT with O'Brien-Fleming error spending
llr_values_obf, A_values_obf, B_values_obf, results_obf = sprt_test_error_spending(
    p0, p1, cumulative_conversions_b, cumulative_visitors_b,
    alpha=0.05, beta=0.2, intervals=check_intervals,
    error_spending_function=o_brien_fleming_spending
)

st.write("\nSPRT Results with O'Brien-Fleming Error Spending:")
for i in range(len(results_obf)):
    st.write(f"Interval {i+1}: LLR = {llr_values_obf[i:.4f]}, Upper A = {A_values_obf[i:.4f]}, Lower B = {B_values_obf[i:.4f]}, Result = {results_obf[i]}")
    if "Reject H0" in results_obf[i] or "Fail to reject H0" in results_obf[i]:
        break

# Run SPRT with Pocock error spending
llr_values_pocock, A_values_pocock, B_values_pocock, results_pocock = sprt_test_error_spending(
    p0, p1, cumulative_conversions_b, cumulative_visitors_b,
    alpha=0.05, beta=0.2, intervals=check_intervals,
    error_spending_function=pocock_spending
)

st.write("\nSPRT Results with Pocock Error Spending:")
for i in range(len(results_pocock)):
    st.write(f"Interval {i+1}: LLR = {llr_values_pocock[i:.4f]}, Upper A = {A_values_pocock[i:.4f]}, Lower B = {B_values_pocock[i:.4f]}, Result = {results_pocock[i]}")
    if "Reject H0" in results_pocock[i] or "Fail to reject H0" in results_pocock[i]:
        break

# Run SPRT without error spending (for comparison)
llr_values_no_spending, A_values_no_spending, B_values_no_spending, results_no_spending = sprt_test_error_spending(
    p0, p1, cumulative_conversions_b, cumulative_visitors_b,
    alpha=0.05, beta=0.2, intervals=check_intervals,
    error_spending_function=None
)

st.write("\nSPRT Results without Error Spending:")
for i in range(len(results_no_spending)):
    st.write(f"Interval {i+1}: LLR = {llr_values_no_spending[i:.4f]}, Upper A = {A_values_no_spending[i:.4f]}, Lower B = {B_values_no_spending[i:.4f]}, Result = {results_no_spending[i]}")
    if "Reject H0" in results_no_spending[i] or "Fail to reject H0" in results_no_spending[i]:
        break

## -- MAIN CODE --
# --- Configuration ---
# Define your control group data (A) - these could also be user inputs
# Example: Replace with actual observed data or user inputs
CONTROL_VISITORS = 1000
CONTROL_CONVERSIONS = 50
P0 = CONTROL_CONVERSIONS / CONTROL_VISITORS if CONTROL_VISITORS > 0 else 0.0

# --- Database Connection (Conceptual) ---
# Use st.connection to manage database access securely
# Replace 'your_database_name' with the actual connection name configured in .streamlit/secrets.toml
# Example: conn = st.connection('postgresql', type='sql')
# conn = st.connection('sqlite', type='sql') # SQLite example - persistence needs careful setup on Cloud
# For simplicity in this example, we'll simulate data storage using session_state
# but remember THIS IS NOT FOR PERSISTENCE ACROSS SESSIONS/RESTARTS on Streamlit Cloud.
# A real implementation needs persistent storage.

# Initialize session state for data if not already present (for simulation only)
if 'cumulative_data_b' not in st.session_state:
    # In a real app, this would attempt to load data from your persistent storage
    st.session_state.cumulative_data_b = pd.DataFrame(columns=['Interval', 'Date', 'Visitors_B', 'Conversions_B'])

# --- Helper Functions for (Simulated) Data Storage ---
# In a real app, these functions would interact with your chosen database/persistent storage

def load_cumulative_data():
    # In a real app, connect to DB and load data
    # Example using st.connection:
    # df = conn.query("SELECT * FROM cumulative_experiment_data ORDER BY Interval;")
    return st.session_state.cumulative_data_b # Simulation

def save_cumulative_data(new_data_point):
    # In a real app, connect to DB and insert new data
    # Example using st.connection:
    # conn.execute("INSERT INTO cumulative_experiment_data (Interval, Date, Visitors_B, Conversions_B) VALUES (?, ?, ?, ?)",
    #              (new_data_point['Interval'], new_data_point['Date'], new_data_point['Visitors_B'], new_data_point['Conversions_B']))
    df = load_cumulative_data()
    st.session_state.cumulative_data_b = pd.concat([df, pd.DataFrame([new_data_point])], ignore_index=True) # Simulation

def clear_cumulative_data():
    # In a real app, connect to DB and delete all data
    # Example using st.connection:
    # conn.execute("DELETE FROM cumulative_experiment_data;")
    st.session_state.cumulative_data_b = pd.DataFrame(columns=['Interval', 'Date', 'Visitors_B', 'Conversions_B']) # Simulation


# --- Streamlit App Layout ---

st.title("Sequential A/B Testing with Error Spending")

st.sidebar.header("Experiment Settings (Group A as Control)")
st.sidebar.info(f"Using Control Group A Data: Visitors = {CONTROL_VISITORS}, Conversions = {CONTROL_CONVERSIONS}, Rate (p0) = {P0:.4f}")

# User input for the alternative hypothesis rate (p1)
# You might want to calculate this based on an expected lift over p0 instead
P1 = st.sidebar.number_input("Target Conversion Rate (p1)", min_value=0.0, max_value=1.0, value=P0 * 1.10, format="%.4f")
st.sidebar.info(f"Assuming Alternative Rate (p1) = {P1:.4f} (e.g., {((P1/P0)-1)*100:.1f}% lift)")

# Ensure p1 is different from p0 for LLR calculation
if P1 == P0:
     st.sidebar.warning("p1 should be different from p0 for meaningful LLR calculation. Consider a target lift.")
     P1 = P0 * 1.0001 # Slightly adjust to avoid log(1) issues if truly identical

ALPHA = st.sidebar.slider("Overall Alpha (Type I Error)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
BETA = st.sidebar.slider("Overall Beta (Type II Error)", min_value=0.05, max_value=0.30, value=0.20, step=0.01) # Common beta is 0.2 (80% power)

st.sidebar.header("Error Spending Function")
error_spending_option = st.sidebar.selectbox(
    "Choose Error Spending Function:",
    options=["None", "O'Brien-Fleming", "Pocock", "Haybittle-Peto"]
)

error_spending_func = None
if error_spending_option == "O'Brien-Fleming":
    error_spending_func = o_brien_fleming_spending
elif error_spending_option == "Pocock":
    error_spending_func = pocock_spending
elif error_spending_option == "Haybittle-Peto":
    error_spending_func = haybittle_peto_spending


# --- Data Entry Section ---

st.header("Enter Cumulative Group B Data")
st.info("Enter the *cumulative* number of visitors and conversions for Group B up to the current analysis interval (e.g., end of day/week).")

with st.form(key='data_entry_form'):
    current_interval = len(load_cumulative_data()) + 1
    st.write(f"Entering data for Interval {current_interval}")

    entry_date = st.date_input("Date for this entry")
    visitors_b_entry = st.number_input(f"Cumulative Visitors for Group B (up to {entry_date})", min_value=0, step=1)
    conversions_b_entry = st.number_input(f"Cumulative Conversions for Group B (up to {entry_date})", min_value=0, step=1)

    submit_button = st.form_submit_button(label='Add Data Point')

    if submit_button:
        if visitors_b_entry < conversions_b_entry:
            st.error("Cumulative visitors must be greater than or equal to cumulative conversions.")
        else:
             # Basic check to ensure data is increasing (cumulative)
            existing_data = load_cumulative_data()
            if not existing_data.empty:
                last_visitors = existing_data['Visitors_B'].iloc[-1]
                last_conversions = existing_data['Conversions_B'].iloc[-1]
                if visitors_b_entry < last_visitors or conversions_b_entry < last_conversions:
                    st.error("Cumulative data must be non-decreasing. Please check your entry.")
                else:
                    new_data = {
                        'Interval': current_interval,
                        'Date': entry_date,
                        'Visitors_B': visitors_b_entry,
                        'Conversions_B': conversions_b_entry
                    }
                    save_cumulative_data(new_data)
                    st.success(f"Data for Interval {current_interval} added.")
                    # Rerun to clear the form fields and update data display
                    st.rerun()
            else:
                # First data point
                new_data = {
                    'Interval': current_interval,
                    'Date': entry_date,
                    'Visitors_B': visitors_b_entry,
                    'Conversions_B': conversions_b_entry
                }
                save_cumulative_data(new_data)
                st.success(f"Data for Interval {current_interval} added.")
                st.rerun()


# --- Display Cumulative Data ---
st.header("Cumulative Data Entered (Group B)")
cumulative_data_df = load_cumulative_data()

if cumulative_data_df.empty:
    st.info("No data points entered yet.")
else:
    st.dataframe(cumulative_data_df)

# --- Run Analysis Section ---
st.header("Run Sequential Analysis")

if st.button("Run SPRT Analysis"):
    if cumulative_data_df.empty:
        st.warning("Please enter cumulative data points to run the analysis.")
    elif P1 <= P0:
         st.warning("Target p1 must be greater than p0 for a one-sided test for improvement.")
    else:
        st.info("Running analysis...")

        # Extract cumulative lists for the SPRT function
        cumulative_conversions_b_list = cumulative_data_df['Conversions_B'].tolist()
        cumulative_visitors_b_list = cumulative_data_df['Visitors_B'].tolist()
        intervals_to_check = list(range(len(cumulative_visitors_b_list))) # Check at every entered data point

        # Perform the SPRT test
        llr_values, A_values, B_values, results = sprt_test_error_spending(
            P0, P1, cumulative_conversions_b_list, cumulative_visitors_b_list,
            alpha=ALPHA, beta=BETA, intervals=intervals_to_check,
            error_spending_function=error_spending_func
        )

        st.subheader("Analysis Results at Each Interval")
        analysis_summary = []
        decision_made = False

        for i in range(len(results)):
            interval_num = intervals_to_check[i] + 1
            summary = {
                "Interval": interval_num,
                "Visitors_B": cumulative_visitors_b_list[i],
                "Conversions_B": cumulative_conversions_b_list[i],
                "LLR": llr_values[i],
                "Upper_Boundary_A": A_values[i],
                "Lower_Boundary_B": B_values[i],
                "Decision": results[i]
            }
            analysis_summary.append(summary)

            if "Reject H0" in results[i] or "Fail to reject H0" in results[i]:
                decision_made = True
                break # Stop displaying if a decision was made

        analysis_df = pd.DataFrame(analysis_summary)
        st.dataframe(analysis_df)

        if decision_made:
            st.success(f"Experiment concluded at Interval {analysis_summary[-1]['Interval']}: {analysis_summary[-1]['Decision']}")
        else:
            st.info("No conclusive result yet. Continue testing.")


        # --- Optional: Plotting ---
        st.subheader("LLR and Boundaries Over Intervals")
        plot_df = pd.DataFrame({
            'Interval': [a['Interval'] for a in analysis_summary],
            'LLR': [a['LLR'] for a in analysis_summary],
            'Upper Boundary (A)': [a['Upper_Boundary_A'] for a in analysis_summary],
            'Lower Boundary (B)': [a['Lower_Boundary_B'] for a in analysis_summary]
        })

        st.line_chart(plot_df.set_index('Interval'))


# --- Data Management ---
st.header("Data Management")
if st.button("Clear All Cumulative Data"):
    clear_cumulative_data()
    st.warning("All cumulative data has been cleared.")
    st.rerun()
