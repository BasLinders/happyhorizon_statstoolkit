import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Constants and Defaults ---
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
DEFAULT_WINRATE = 0.01
DEFAULT_SIGMOID_K = 0.03  # Sigmoid growth rate
DEFAULT_SIGMOID_X0 = 20  # Diminishing returns midpoint
DEFAULT_ITERATIONS = 10000
# Uplift scaling factors mapped to variability levels
UPLIFT_SCALING_FACTORS = {
    "Low": 0.05,
    "Medium": 0.10,
    "High": 0.15
}
DEFAULT_VARIABILITY = "Medium"  # Default variability level

# --- Helper Functions ---

def calculate_mde(cr, visitors, alpha=DEFAULT_ALPHA, power=DEFAULT_POWER):
    """Calculates the Minimum Detectable Effect (MDE)."""
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(cr * (1 - cr) * 2 / (visitors / 2))

def sigmoid(x, x0=DEFAULT_SIGMOID_X0, k=DEFAULT_SIGMOID_K):
    """Calculates the sigmoid function (flipped for diminishing returns)."""
    return 1 - (1 / (1 + np.exp(-k * (x - x0))))

def monte_carlo_simulation(cr_base, n_experiments_range, winrate, mde_min, mde_max, sigmoid_k, sigmoid_x0, iterations=DEFAULT_ITERATIONS, uplift_scaling_factor=0.25):
    """Performs the Monte Carlo simulation."""
    results = []

    for n_experiments in n_experiments_range:
        simulated_uplifts_min = []
        simulated_uplifts_max = []

        for _ in range(iterations):
            # Apply diminishing returns
            factor = sigmoid(n_experiments, sigmoid_x0, sigmoid_k)

            # Calculate mean relative uplifts
            mean_relative_uplift_min = winrate * (mde_min / cr_base) if cr_base > 0 else 0
            mean_relative_uplift_max = winrate * (mde_max / cr_base) if cr_base > 0 else 0

            # Calculate standard deviations
            sd_uplift_min = mde_min * winrate * uplift_scaling_factor # Use passed-in value
            sd_uplift_max = mde_max * winrate * uplift_scaling_factor

            # Sample *absolute* uplifts from normal distributions
            sampled_absolute_uplift_min = np.random.normal(mde_min * winrate, sd_uplift_min) * factor
            sampled_absolute_uplift_max = np.random.normal(mde_max * winrate, sd_uplift_max) * factor

            # Convert to relative uplift
            sampled_relative_uplift_min = sampled_absolute_uplift_min / cr_base if cr_base > 0 else 0
            sampled_relative_uplift_max = sampled_absolute_uplift_max / cr_base if cr_base > 0 else 0

            # Apply compounding
            cr_current_min = cr_base * (1 + sampled_relative_uplift_min) ** n_experiments
            cr_current_max = cr_base * (1 + sampled_relative_uplift_max) ** n_experiments

            # Calculate cumulative uplift
            cumulative_uplift_min = (cr_current_min / cr_base) - 1
            cumulative_uplift_max = (cr_current_max / cr_base) - 1

            simulated_uplifts_min.append(cumulative_uplift_min)
            simulated_uplifts_max.append(cumulative_uplift_max)

        results.append({
            "Experiments": n_experiments,
            "Min_Mean_Uplift": round(np.mean(simulated_uplifts_min) * 100, 2),
            "Max_Mean_Uplift": round(np.mean(simulated_uplifts_max) * 100, 2),
            "Min_Lower_Bound": round(np.percentile(simulated_uplifts_min, 5) * 100, 2),
            "Min_Upper_Bound": round(np.percentile(simulated_uplifts_min, 95) * 100, 2),
            "Max_Lower_Bound": round(np.percentile(simulated_uplifts_max, 5) * 100, 2),
            "Max_Upper_Bound": round(np.percentile(simulated_uplifts_max, 95) * 100, 2),
        })

    return pd.DataFrame(results)

def plot_simulation_results(simulation_df, n_experiments_max):
    """Plots the simulation results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(simulation_df["Experiments"], simulation_df["Min_Mean_Uplift"], label="Min Mean Uplift", color="red")
    ax.plot(simulation_df["Experiments"], simulation_df["Max_Mean_Uplift"], label="Max Mean Uplift", color="green")
    ax.fill_between(simulation_df["Experiments"], simulation_df["Min_Lower_Bound"], simulation_df["Min_Upper_Bound"], color="red", alpha=0.2, label="Min 90% CI")
    ax.fill_between(simulation_df["Experiments"], simulation_df["Max_Lower_Bound"], simulation_df["Max_Upper_Bound"], color="green", alpha=0.2, label="Max 90% CI")
    ax.set_xlabel("Number of Experiments")
    ax.set_ylabel("Cumulative Uplift (%)")
    ax.set_title(f"Experiment and Uplift Forecast for {n_experiments_max} Experiments")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- Streamlit App ---

def run():
    st.set_page_config(page_title="Annual Compound Growth Calculator", page_icon="🔢")

    # Initialize session state
    st.session_state.setdefault("visitors_base", 0)
    st.session_state.setdefault("conv_base", 0)
    st.session_state.setdefault("winrate", DEFAULT_WINRATE)
    st.session_state.setdefault("used_months", 1)
    st.session_state.setdefault("n_experiments_max", 1)
    st.session_state.setdefault("alpha", DEFAULT_ALPHA)
    st.session_state.setdefault("power", DEFAULT_POWER)
    st.session_state.setdefault("sigmoid_k", DEFAULT_SIGMOID_K)
    st.session_state.setdefault("sigmoid_x0", DEFAULT_SIGMOID_X0)
    st.session_state.setdefault("variability_level", DEFAULT_VARIABILITY)

    st.title("Annual Compound Growth Calculator")
    st.write("""
    This calculator will estimate annual compound growth for experimentation in worst- and best-case scenarios.
    Each table row corresponds to the cumulative impact of conducting more experiments.
    The confidence intervals per row provide the minimum (lower bound) and maximum (upper bound) uplifts in the 95% simulations conducted by the tool.

    This demonstrates the power of running more experiments over time.

    Choose a representative time frame for your business goals and enter the respective visitors and conversions for that time frame. 
    The tool will then return compound effects for the chosen range of experiments, projected over the course of 12 months.

    The resulting table provides powerful insights into the expected performance of your experiments. You can use it to:

    - Estimate the impact of running multiple experiments.
    - Communicate realistic expectations (with uncertainty ranges).
    - Make informed decisions about whether the potential rewards justify the effort.
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Baseline Data")
        used_months = st.number_input("Months of data collection?", min_value=1, max_value=12, step=1, value=st.session_state.get("used_months",1), key="used_months")
        visitors_base = st.number_input("Visitors?", min_value=0, step=1, value=st.session_state.get("visitors_base",0), key="visitors_base")
        conv_base = st.number_input("Conversions?", min_value=0, step=1, value=st.session_state.get("conv_base",0), key="conv_base")

    with col2:
        st.write("### Experimentation Program")
        winrate = st.number_input(
            "Win Rate (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=max(0.0, float(st.session_state.get("winrate", DEFAULT_WINRATE))) * 100,
            key = "winrate_input"
        ) / 100

        n_experiments_max = st.number_input(
            "Max Experiments",
            min_value=1,
            step=1,
            value=max(1, int(st.session_state.get("n_experiments_max", 1))),
            key="n_experiments_max"
        )
        alpha = st.number_input("Significance Level (alpha)", min_value=0.001, max_value=0.999, value=st.session_state.get("alpha", DEFAULT_ALPHA), step=0.01,key="alpha")
        power = st.number_input("Desired Power (1 - beta)", min_value=0.01, max_value=0.99, value=st.session_state.get("power", DEFAULT_POWER), step=0.01,key="power")
        sigmoid_k = st.number_input(
            "Diminishing Returns Curve Steepness (k)",
            min_value=0.001,
            max_value=0.5,
            value=st.session_state.get("sigmoid_k", DEFAULT_SIGMOID_K),
            step=0.001,
            help="Controls how quickly diminishing returns set in (higher = steeper).",
            key="sigmoid_k"
        )
        sigmoid_x0 = st.number_input(
            "Diminishing Returns Curve Midpoint (x0)",
            min_value=0,
            max_value=100,
            value=st.session_state.get("sigmoid_x0", DEFAULT_SIGMOID_X0),
            step=1,
            help="Number of experiments at which diminishing returns are strongest.",
            key="sigmoid_x0"
        )

        variability_level = st.selectbox(
            "Uplift Variability Level",
            options=list(UPLIFT_SCALING_FACTORS.keys()),
            index=list(UPLIFT_SCALING_FACTORS.keys()).index(st.session_state.get("variability_level", DEFAULT_VARIABILITY)),  # Set default and get from session state
            help="Controls the width of the confidence intervals. Lower variability = narrower intervals.",
            key = "variability_level"
        )

    if st.button("Calculate Projected Uplift"):
        if used_months > 0 and visitors_base > 0 and conv_base > 0 and n_experiments_max > 0:
            # Annualization
            v_twelve = round((visitors_base / used_months) * 12)
            c_twelve = round((conv_base / used_months) * 12)
            conv_base = c_twelve
            visitors_base = v_twelve

            # Baseline conversion rate
            cr_base = conv_base / visitors_base if visitors_base > 0 else 0

            # Calculate cr_min and cr_max (heuristic)
            cr_min = cr_base
            cr_max = cr_base * (1 + np.log1p(1.5))

            # Calculate mde_min and mde_max
            mde_min = calculate_mde(cr_min, visitors_base, alpha, power)
            mde_max = calculate_mde(cr_max, visitors_base, alpha, power)

            # Calculate relative mde_min and mde_max.
            relative_mde_min = (mde_min / cr_min) * 100 if cr_min > 0 else 0
            relative_mde_max = (mde_max / cr_max) * 100 if cr_max > 0 else 0


            st.write("### Computed Statistics")
            st.write(f"Baseline Conversion Rate: {cr_base:.4f}")
            st.write(f"Hypothetical Minimum Conversion Rate: {cr_min:.4f}")
            st.write(f"Hypothetical Maximum Conversion Rate: {cr_max:.4f}")
            st.write(f"Hypothetical Minimum Absolute MDE: {mde_min:.6f}")
            st.write(f"Hypothetical Minimum Relative MDE: {relative_mde_min:.2f}%")
            st.write(f"Hypothetical Maximum Absolute MDE: {mde_max:.6f}")
            st.write(f"Hypothetical Maximum Relative MDE: {relative_mde_max:.2f}%")

            # Get the scaling factor based on user selection
            uplift_scaling_factor = UPLIFT_SCALING_FACTORS[variability_level]


            # --- Monte Carlo Simulation ---
            n_experiments_range = list(range(1, n_experiments_max + 1))
            simulation_df = monte_carlo_simulation(
                cr_base,
                n_experiments_range,
                winrate,
                mde_min,
                mde_max,
                sigmoid_k,
                sigmoid_x0,
                iterations=DEFAULT_ITERATIONS,
                uplift_scaling_factor=uplift_scaling_factor, #Pass down scaling factor
            )

            st.write("### Simulation Results")
            st.dataframe(simulation_df)
            plot_simulation_results(simulation_df, n_experiments_max)

        else:
            st.write("")
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()