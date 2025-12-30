import streamlit as st
import pandas as pd
import numpy as np
import uuid
from st_supabase_connection import SupabaseConnection

# --- 1. CONFIGURATION & CONNECTION ---
st.set_page_config(page_title="Sequential Analysis", layout="wide")

# Initialize Supabase Connection
conn = st.connection("supabase", type=SupabaseConnection)

# --- 2. STATISTICAL FUNCTIONS ---

def calculate_wald_boundaries(alpha, beta):
    """
    Calculates the upper (rejection) and lower (futility) boundaries
    for Wald's Sequential Probability Ratio Test (SPRT).
    """
    # Boundary A (Upper): Log-likelihood ratio threshold for rejecting H0 (Success)
    # Approximation: A = (1 - beta) / alpha
    upper = np.log((1 - beta) / alpha)
    
    # Boundary B (Lower): Log-likelihood ratio threshold for accepting H0 (Futility)
    # Approximation: B = beta / (1 - alpha)
    lower = np.log(beta / (1 - alpha))
    
    return upper, lower

def calculate_llr(p0, p1, conversions, visitors):
    """
    Calculates the cumulative Log Likelihood Ratio for Binomial data.
    """
    if visitors == 0:
        return 0.0
    
    # Clip probabilities to avoid log(0)
    eps = 1e-9
    p0 = np.clip(p0, eps, 1 - eps)
    p1 = np.clip(p1, eps, 1 - eps)
    
    # LLR = x * log(p1/p0) + (n-x) * log((1-p1)/(1-p0))
    term1 = conversions * np.log(p1 / p0)
    term2 = (visitors - conversions) * np.log((1 - p1) / (1 - p0))
    
    return term1 + term2

# --- 3. DATABASE FUNCTIONS ---
def get_experiment_params(experiment_id):
    """Fetch setup parameters (p0, p1, alpha, beta) for an ID."""
    try:
        response = conn.table("experiment_params").select("*").eq("experiment_id", experiment_id).execute()
        if len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        st.error(f"Error fetching params: {e}")
        return None

def save_experiment_params(experiment_id, p0, p1, alpha, beta, max_visitors, test_type):
    """Save the immutable rules of the experiment."""
    try:
        data = {
            "experiment_id": experiment_id,
            "p0": float(p0),
            "p1": float(p1),
            "alpha": float(alpha),
            "beta": float(beta),
            "max_visitors": int(max_visitors),
            "test_type": str(test_type)
        }
        conn.table("experiment_params").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error creating experiment: {e}")
        return False
        
def get_experiment_data(experiment_id):
    try:
        response = conn.table("msprt_data").select("*").eq("experiment_id", experiment_id).order("measurement_date").execute()
        if len(response.data) > 0:
            df = pd.DataFrame(response.data)
            df['measurement_date'] = pd.to_datetime(df['measurement_date']).dt.date
            # Handle potential None values for control columns if loading old data
            df['visitors'] = df['visitors'].fillna(0).astype(int)
            df['conversions'] = df['conversions'].fillna(0).astype(int)
            # Check if columns exist (for backward compatibility)
            if 'visitors_control' in df.columns:
                df['visitors_control'] = df['visitors_control'].fillna(0).astype(int)
                df['conversions_control'] = df['conversions_control'].fillna(0).astype(int)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def save_data_point(experiment_id, date, v_variant, c_variant, v_control=0, c_control=0):
    """Insert a new cumulative data point."""
    try:
        data = {
            "experiment_id": experiment_id,
            "measurement_date": str(date),
            "visitors": int(v_variant),
            "conversions": int(c_variant),
            "visitors_control": int(v_control),
            "conversions_control": int(c_control)
        }
        conn.table("msprt_data").insert(data).execute()
        st.toast("Data point saved successfully!", icon="✅")
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

# --- 4. UI LOGIC ---
def run():
    st.title("Sequential Experiment Analysis (SPRT)")
    st.markdown("""
    ### Faster A/B Testing with Sequential Analysis
    Standard A/B tests require you to wait for a fixed sample size to avoid "peeking" errors. 
    **This tool is different.** It uses **Sequential Probability Ratio Testing (SPRT)**, allowing you to update data and check results **any time** without invalidating your statistics.
    """)

    with st.expander("Benefits of SPRT", expanded=False):
        st.markdown("""
        #### Benefits of SPRT
        Compared to fixed-horizon testing, SPRT has certain advantages.
        * **Stop Winners Early:** Deploy successful features days or weeks faster.
        * **Cut Losers Fast:** Identify "futility" (no chance of winning) early to save traffic and/or money.
        * **Rigorous:** Mathematically valid stopping rules, unlike standard z-tests ('peeking' is valid).
        """)
    with st.expander("When to use SPRT", expanded=False):
        st.markdown("""
        ### When to use SPRT
        Sequential probability ratio testing is an agile tool. You should use it when:
        * **Safety is a concern:** You want to kill a 'losing' experiment immediately if it's tanking metrics.
        * **The observed effect is huge:** If the new feature is a massive success, SPRT will let you ship it in (for example) 3 days instead of 14.
        * **The cost of testing is high:** If every user in the experiment costs money, stopping early saves budget.

        ### When to use fixed-horizon testing
        * If you have a **strict deadline**.
        * SPRT has a **slightly wider confidence interval** and will thus overestimate the treatment effect.
        * If you're looking for a **tiny lift** (e.g. 0.5% increase) it might take longer to reach a conclusion than a fixed-horizon test as in this case, SPRT has generally less statistical power.
        * If stakeholders are better aligned with clear deadlines and mid- to long-term planning of experiments.
        """)
    with st.expander("How to use SPRT", expanded=False):
        st.markdown("""
        #### How to use SPRT
        1.  **Start New:** Generate a unique ID and define your success metrics (Alpha, Beta, MDE). 
            * *Note: These are locked once the test starts to ensure integrity.*
        2.  **Update Regularly:** Come back daily/weekly to input your **cumulative** data.
        3.  **Check the Graph:** 
            * **Upper Limit:** Success! (Reject Null)
            * **Lower Limit:** Futility/Failure. (Accept Null)
        
        > * **Important:** Data is stored for **42 days (6 weeks)** and then automatically deleted.
        > * **Save your Experiment ID!** It is the only key to retrieve your data.
        """)
    
    # Initialize Session State for locking
    if 'params_locked' not in st.session_state: st.session_state['params_locked'] = False
    if 'fetched_params' not in st.session_state: st.session_state['fetched_params'] = {}

    # Logic: Use fetched values if locked, otherwise use defaults
    defaults = st.session_state.get('fetched_params', {})
    is_locked = st.session_state.get('params_locked', False)
    
# --- SIDEBAR: SETUP & LOADING ---
    with st.sidebar:
        st.header("1. Experiment Setup")

        # 1. Test Type Selection (Synced with DB)
        options = ["One-sample (fixed baseline)", "Two-sample (concurrent control/variant)"]
        saved_type = defaults.get('test_type', options[0])
        try:
            default_index = options.index(saved_type)
        except ValueError:
            default_index = 0

        test_type = st.radio("Test format", options, index=default_index, disabled=is_locked)
        
        # Dynamic Labels
        if test_type == "One-sample (fixed baseline)":
            p0_label = "Baseline CR (p0)"
            p0_help = "The fixed historical conversion rate (CR) you want to beat."
        else:
            p0_label = "Estimated baseline CR (p0)"
            p0_help = "Used only to calculate sample size estimates. The actual Control CR will be measured live."

        # 2. Mode Selection
        mode = st.radio("Mode", ["Load Existing", "Start New"], label_visibility="collapsed")
        
        # 3. ID Management
        if mode == "Start New":
            if st.button("Generate New ID"):
                st.session_state['exp_id'] = str(uuid.uuid4())
                st.session_state['params_locked'] = False
                st.session_state['fetched_params'] = {}
                st.rerun()
            
            if st.session_state.get('exp_id'):
                st.success(f"New ID: {st.session_state['exp_id']}")
        
        else: # Load Existing
            input_id = st.text_input("Paste Experiment UUID")
            if st.button("Load"):
                if input_id:
                    st.session_state['exp_id'] = input_id
                    params = get_experiment_params(input_id)
                    if params:
                        st.session_state['fetched_params'] = params
                        st.session_state['params_locked'] = True
                        st.toast("Parameters loaded and locked!", icon="🔒")
                    else:
                        st.error("Experiment ID not found or no parameters set.")
                    st.rerun()

        st.divider()

        # --- PARAMETER INPUTS ---
        st.subheader("2. Test Parameters")

        if is_locked:
            st.info("Parameters are locked for this ID.")

        with st.form("setup_form"):
            p0_val = float(defaults.get('p0', 0.10))
            p1_val = float(defaults.get('p1', 0.12))
            alpha_val = float(defaults.get('alpha', 0.05))
            beta_val = float(defaults.get('beta', 0.20))
            max_visitors_val = int(defaults.get('max_visitors', 10000))

            p0_param = st.number_input(p0_label, value=p0_val, format="%.4f", disabled=is_locked, help=p0_help)
            p1_param = st.number_input("Target CR (p1)", value=p1_val, format="%.4f", disabled=is_locked, 
                                     help="Set this to the Minimum Effect Size.")
            max_visitors = st.number_input("Max Visitors (Safety Cap)", value=max_visitors_val, step=100, disabled=is_locked)
            
            c1, c2 = st.columns(2)
            alpha = c1.number_input("Alpha", value=alpha_val, step=0.01, disabled=is_locked)
            beta = c2.number_input("Beta", value=beta_val, step=0.01, disabled=is_locked)

            if not is_locked:
                submitted = st.form_submit_button("Start & Lock Experiment")
                if submitted:
                    if not st.session_state.get('exp_id'):
                        st.error("Generate an ID first!")
                    elif p1_param <= p0_param:
                        st.error("p1 must be > p0")
                    else:
                        saved = save_experiment_params(st.session_state['exp_id'], p0_param, p1_param, alpha, beta, max_visitors, test_type=test_type)
                        if saved:
                            st.session_state['params_locked'] = True
                            st.session_state['fetched_params'] = {
                                'p0': p0_param, 'p1': p1_param, 'alpha': alpha, 'beta': beta, 
                                'max_visitors': max_visitors, 'test_type': test_type
                            }
                            st.rerun()
            else:
                st.form_submit_button("Parameters Locked", disabled=True)

    # --- MAIN PAGE CONTENT ---
    exp_id = st.session_state.get('exp_id')
    
    if not exp_id or not st.session_state.get('params_locked'):
        st.info("**To Begin:** Select 'Start New' to generate an ID and lock your parameters.")
        st.stop()

    st.markdown(f"### Experiment: `{exp_id}`")
    
    df = get_experiment_data(exp_id)
    
    # Use the test type from the locked params, fallback to One-sample
    current_test_type = st.session_state['fetched_params'].get('test_type', "One-sample (fixed baseline)")
    
    # --- DATA ENTRY ---
    st.subheader("Update Data")

    # Correct form name here
    with st.form("entry_form"):
        # Date is outside columns
        d_date = st.date_input("Date")
        
        prev_vis = int(df.iloc[-1]['visitors']) if not df.empty else 0
        prev_conv = int(df.iloc[-1]['conversions']) if not df.empty else 0
        
        # LOGIC FORK: Layouts
        if current_test_type == "Two-sample (concurrent control/variant)":
            st.divider()
            
            # Row 1: Control
            st.markdown("### Control Group")
            c_a1, c_a2 = st.columns(2)
            d_vis_c = c_a1.number_input("Control Visitors", min_value=0)
            d_conv_c = c_a2.number_input("Control Conversions", min_value=0)
            
            # Row 2: Variant
            st.markdown("### Variant Group")
            c_b1, c_b2 = st.columns(2)
            d_vis = c_b1.number_input("Variant Visitors", min_value=0)
            d_conv = c_b2.number_input("Variant Conversions", min_value=0)
            
            save_v_c, save_c_c = d_vis_c, d_conv_c
        
        else:
            st.divider()
            st.markdown("### Variant Data")
            c1, c2 = st.columns(2)
            d_vis = c1.number_input(f"Cumulative Visitors (Prev: {prev_vis})", min_value=prev_vis, value=prev_vis)
            d_conv = c2.number_input(f"Cumulative Conversions (Prev: {prev_conv})", min_value=prev_conv, value=prev_conv)
            save_v_c, save_c_c = 0, 0
        
        st.divider()
        if st.form_submit_button("Add Data Point"):
            if d_vis < d_conv:
                st.error("Visitors cannot be less than conversions.")
            else:
                save_data_point(exp_id, d_date, d_vis, d_conv, v_control=save_v_c, c_control=save_c_c)
                st.rerun()

    # --- ANALYSIS SECTION ---
    if not df.empty:
        st.divider()
        st.subheader("Sequential Analysis")

        if current_test_type == "Two-sample (concurrent control/variant)":
            st.info("**Data collection mode**")
            st.write("Two-Sample data is being saved securely to the database.")
            st.warning("Analysis for Two-Sample tests is currently under construction.")

            with st.expander("View Raw Data"):
                st.dataframe(df)
        else: 
            # 1. Use the locked parameters
            upper_bound, lower_bound = calculate_wald_boundaries(alpha, beta)
            
            df['llr'] = df.apply(lambda row: calculate_llr(p0_param, p1_param, row['conversions'], row['visitors']), axis=1)
            
            latest_llr = df.iloc[-1]['llr']
            latest_vis = df.iloc[-1]['visitors']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Lower Bound (Stop for Futility)", f"{lower_bound:.2f}")
            col2.metric("Current LLR", f"{latest_llr:.2f}", delta_color="off")
            col3.metric("Upper Bound (Stop for Success)", f"{upper_bound:.2f}")
            
            # 3. Decision Logic
            latest_vis = df.iloc[-1]['visitors']
            if latest_vis > 0:
                latest_cr = df.iloc[-1]['conversions'] / latest_vis
            else:
                latest_cr = 0.0
            
            if latest_llr > upper_bound:
                st.success(f"### Result: SIGNIFICANT POSITIVE (Reject H0)")
                st.write(f"The Variant is statistically superior. You can stop the test early at {latest_vis} visitors.")
            elif latest_llr < lower_bound:
                if latest_cr < p0_param:
                    st.error(f"### Result: SIGNIFICANT NEGATIVE")
                    st.write(f"The Variant is performing **worse** than Control. Stop immediately.")
                else:
                    st.error(f"### Result: FUTILITY (Accept H0)")
                    st.write(f"The Variant is unlikely to reach the target.")
            else:
                if latest_vis >= max_visitors:
                    st.write("Maximum sample size reached without a decision.")
                else:
                    st.warning(f"### Result: INCONCLUSIVE")
                    st.write("Continue collecting data. The test has not yet breached a boundary.")
        
            # 4. Visualization
            st.markdown("### Test Trajectory")
            chart_data = df[['visitors', 'llr']].copy()
            chart_data['Upper Boundary'] = upper_bound
            chart_data['Lower Boundary'] = lower_bound
            
            st.line_chart(chart_data.set_index('visitors'), color=["#FF4B4B", "#0000FF", "#0000FF"]) 
            
            with st.expander("View Raw Data"):
                st.dataframe(df.sort_values("measurement_date", ascending=False))
    
    else:
        st.write("Waiting for data entries...")

if __name__ == "__main__":
    run()
