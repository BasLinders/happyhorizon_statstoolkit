import streamlit as st
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import string

st.set_page_config(
    page_title="Sequential Analysis",
    page_icon="",
)

def initialize_session_state():
    st.session_state.setdefault("num_variants", 2)
    st.session_state.setdefault("sequential_params", {})
    st.session_state.setdefault("stage_data", [])
    st.session_state.setdefault("analysis_results", [])
    st.session_state.setdefault("current_stage", 1)
    st.session_state.setdefault("stage_labels", [])
    st.session_state.setdefault("visitor_counts", [0] * 10)  # Initialize with a max of 10 variants
    st.session_state.setdefault("variant_conversions", [0] * 10)

def display_intro():
    st.write("# Sequential Analysis")
    st.markdown("""
        This tool performs sequential analysis for A/B tests, allowing for early stopping.
        Define your experiment stages and input data at each stage to monitor results.
        """)

def get_sequential_parameters():
    st.sidebar.header("Sequential Analysis Parameters")
    st.session_state.sequential_params['num_stages'] = st.sidebar.number_input(
        "Number of Stages", min_value=2, step=1, value=3
    )
    st.session_state.sequential_params['alpha_boundary'] = st.sidebar.number_input(
        "Alpha Boundary per Stage (%)", min_value=0.1, max_value=10.0, value=5.0, step=0.1
    ) / 100
    st.session_state.sequential_params['stopping_rule'] = st.sidebar.selectbox(
        "Stopping Rule", ["Fixed Alpha", "O'Brien-Fleming (approx.)", "Pocock (approx.)"], index=0
    )
    return st.session_state.sequential_params

def get_stage_inputs(stage_num):
    st.subheader(f"Stage {stage_num} Data")
    st.session_state.num_variants = st.number_input(
        "How many variants did your experiment have (including control)?",
        min_value=2, max_value=10, step=1, value=st.session_state.num_variants
    )

    if len(st.session_state.visitor_counts) != st.session_state.num_variants:
        st.session_state.visitor_counts = st.session_state.visitor_counts[:st.session_state.num_variants] + [0] * (st.session_state.num_variants - len(st.session_state.visitor_counts))
    if len(st.session_state.variant_conversions) != st.session_state.num_variants:
        st.session_state.variant_conversions = st.session_state.variant_conversions[:st.session_state.num_variants] + [0] * (st.session_state.num_variants - len(st.session_state.variant_conversions))

    num_variants = st.session_state.num_variants
    visitor_counts_stage = []
    variant_conversions_stage = []
    alphabet = string.ascii_uppercase

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Visitors")
    with col2:
        st.write("### Conversions")

    for i in range(num_variants):
        with col1:
            visitor_counts_stage.append(st.number_input(
                f"How many visitors did variant {alphabet[i]} have?",
                min_value=0, step=1, value=st.session_state.visitor_counts[i], key=f"visitors_{stage_num}_{i}"
            ))
        with col2:
            variant_conversions_stage.append(st.number_input(
                f"How many conversions did variant {alphabet[i]} have?",
                min_value=0, step=1, value=st.session_state.variant_conversions[i], key=f"conversions_{stage_num}_{i}"
            ))

    return visitor_counts_stage, variant_conversions_stage

def get_stage_labels():
    num_stages = st.session_state.sequential_params.get('num_stages', 1)
    st.subheader("Define Stage Labels")
    stage_labels = []
    for i in range(num_stages):
        label = st.text_input(f"Label for Stage {i+1}", f"Stage {i+1}")
        stage_labels.append(label)
    st.session_state.stage_labels = stage_labels
    return stage_labels

def perform_sequential_analysis(stage_data):
    if not stage_data or len(stage_data) < 2:
        st.warning("Please input data for at least two stages to perform sequential analysis.")
        return None

    results = []
    num_variants = st.session_state.num_variants
    all_visitors = np.zeros(num_variants)
    all_conversions = np.zeros(num_variants)

    for i, data in enumerate(stage_data):
        visitors = np.array(data['visitors'])
        conversions = np.array(data['conversions'])
        all_visitors += visitors
        all_conversions += conversions

        # Simplified Log-Likelihood Ratio Test
        control_conversions = all_conversions[0]
        control_visitors = all_visitors[0]

        stage_results = {}
        stage_results['stage'] = i + 1
        stage_results['log_likelihood_ratios'] = {}
        stage_results['p_values'] = {}

        for j in range(1, num_variants):
            variant_conversions = all_conversions[j]
            variant_visitors = all_visitors[j]

            if control_visitors > 0 and variant_visitors > 0:
                p_control = control_conversions / control_visitors
                p_variant = variant_conversions / variant_visitors

                # Log-Likelihood under null hypothesis (assuming equal conversion rates)
                pooled_conversions = control_conversions + variant_conversions
                pooled_visitors = control_visitors + variant_visitors
                p_pooled = pooled_conversions / pooled_visitors if pooled_visitors > 0 else 0

                log_likelihood_null = (control_conversions * np.log(p_pooled + 1e-9) +
                                       (control_visitors - control_conversions) * np.log(1 - p_pooled + 1e-9) +
                                       variant_conversions * np.log(p_pooled + 1e-9) +
                                       (variant_visitors - variant_conversions) * np.log(1 - p_pooled + 1e-9))

                # Log-Likelihood under alternative hypothesis (observed conversion rates)
                log_likelihood_alt = (control_conversions * np.log(p_control + 1e-9) +
                                      (control_visitors - control_conversions) * np.log(1 - p_control + 1e-9) +
                                      variant_conversions * np.log(p_variant + 1e-9) +
                                      (variant_visitors - variant_conversions) * np.log(1 - p_variant + 1e-9))

                likelihood_ratio = 2 * (log_likelihood_alt - log_likelihood_null)
                stage_results['log_likelihood_ratios'][f'Variant {string.ascii_uppercase[j]} vs Control'] = likelihood_ratio
                p_value = 1 - stats.chi2.cdf(likelihood_ratio, 1)  # Assuming 1 degree of freedom
                stage_results['p_values'][f'Variant {string.ascii_uppercase[j]} vs Control'] = p_value
            else:
                stage_results['log_likelihood_ratios'][f'Variant {string.ascii_uppercase[j]} vs Control'] = None
                stage_results['p_values'][f'Variant {string.ascii_uppercase[j]} vs Control'] = None

        results.append(stage_results)
    return results

def plot_stopping_boundaries(analysis_results, sequential_params, stage_labels):
    num_stages = sequential_params.get('num_stages', 1)
    alpha_boundary = sequential_params.get('alpha_boundary', 0.05)

    fig, ax = plt.subplots()
    stages = range(1, len(analysis_results) + 1)

    # Define placeholder boundaries for demonstration
    upper_boundary = [stats.norm.ppf(1 - alpha_boundary / 2)] * num_stages
    lower_boundary = [-stats.norm.ppf(1 - alpha_boundary / 2)] * num_stages

    # Extract relevant statistic (e.g., for the first variant vs control)
    observed_stats = []
    for res in analysis_results:
        if res and 'log_likelihood_ratios' in res and 'Variant B vs Control' in res['log_likelihood_ratios']:
            llr = res['log_likelihood_ratios']['Variant B vs Control']
            observed_stats.append(np.sqrt(llr) if llr is not None and llr >= 0 else np.nan)

    ax.plot(stages, observed_stats, marker='o', label="Observed Statistic (sqrt(LLR))")
    ax.plot(stages, upper_boundary[:len(stages)], linestyle='--', color='red', label="Efficacy Boundary")
    ax.plot(stages, lower_boundary[:len(stages)], linestyle='--', color='blue', label="Futility Boundary")

    ax.set_xlabel("Stage")
    ax.set_ylabel("Statistic Value")
    ax.set_title("Sequential Analysis with Stopping Boundaries")
    ax.set_xticks(stages)
    if stage_labels:
        ax.set_xticklabels(stage_labels[:len(stages)])
    ax.legend()
    st.pyplot(fig)

def run():
    initialize_session_state()
    display_intro()
    get_sequential_parameters()
    num_stages = st.session_state.sequential_params.get('num_stages', 1)

    stage_labels = get_stage_labels()

    if st.button(f"Input Data for Stage {st.session_state.current_stage}") and st.session_state.current_stage <= num_stages:
        visitors, conversions = get_stage_inputs(st.session_state.current_stage)
        st.session_state.stage_data.append({'visitors': visitors, 'conversions': conversions})
        if st.session_state.current_stage < num_stages:
            st.session_state.current_stage += 1
            st.experimental_rerun()
        else:
            st.info("Data input complete. Click 'Run Sequential Analysis'.")

    if st.button("Run Sequential Analysis"):
        if len(st.session_state.stage_data) >= 2:
            st.session_state.analysis_results = perform_sequential_analysis(st.session_state.stage_data)

    if st.session_state.analysis_results:
        st.subheader("Sequential Analysis Results per Stage")
        for result in st.session_state.analysis_results:
            st.write(f"**Stage {result['stage']} ({st.session_state.stage_labels[result['stage']-1] if st.session_state.stage_labels else f'Stage {result['stage']}'})**")
            if 'log_likelihood_ratios' in result:
                st.write("Log-Likelihood Ratios:")
                for key, value in result['log_likelihood_ratios'].items():
                    st.write(f"- {key}: {value:.4f}" if value is not None else f"- {key}: N/A")
            if 'p_values' in result:
                st.write("P-values:")
                for key, value in result['p_values'].items():
                    st.write(f"- {key}: {value:.4f}" if value is not None else f"- {key}: N/A")

        plot_stopping_boundaries(st.session_state.analysis_results, st.session_state.sequential_params, st.session_state.stage_labels)

if __name__ == "__main__":
    run()
