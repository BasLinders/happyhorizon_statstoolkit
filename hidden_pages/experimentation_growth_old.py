import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import binom, norm, beta

st.set_page_config(
    page_title="Annual Compound Growth calculator",
    page_icon="🔢",
)

def run():
    # Initialize session state for inputs
    st.session_state.setdefault("visitors_base", 0)
    st.session_state.setdefault("conv_base", 0)
    st.session_state.setdefault("winrate", 0.01)
    st.session_state.setdefault("used_months", 1)
    st.session_state.setdefault("n_experiments", 0)
    st.session_state.setdefault("n_experiments_max", 1)

    st.title("Annual Compound Growth Calculator")
    """
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
    """

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Baseline data")
        st.session_state.used_months = st.number_input("Over how many months is the data below been collected?", min_value=1, max_value=12, step=1, value=st.session_state.used_months)
        st.session_state.visitors_base = st.number_input("How many visitors should be used in the estimate?", min_value=0, step=1, value=st.session_state.visitors_base)
        st.session_state.conv_base = st.number_input("How many conversions should be used in the estimate?", min_value=0, step=1, value=st.session_state.conv_base)
    with col2:
        st.write("### Input for estimation")
        st.session_state.winrate = st.number_input(
            "What is the desired proportion of wins overall?",
            min_value=0.01,
            step=0.01,
            value=max(0.01, float(st.session_state.get("winrate", 0.01)))
        )
        st.session_state.n_experiments_max = st.number_input(
            "Max amount of experiments in the estimate?", 
            min_value = 1, 
            step=1, 
            value=max(1, int(st.session_state.get("n_experiments_max", 1)))
        )

    # Variables for computation
    used_months = st.session_state.used_months
    visitors_base = st.session_state.visitors_base
    conv_base = st.session_state.conv_base
    winrate = st.session_state.winrate
    n_experiments_max = st.session_state.n_experiments_max
    n_experiments = n_experiments_max

    if st.button("Calculate my test results"):    
        # Check if visitors and conversions are > 0
        if used_months > 0 and visitors_base > 0 and conv_base > 0 and n_experiments_max > 0:

            # Calculated or hardcoded statistics
            n_experiments_range = list(range(1, n_experiments_max + 1))
            v_twelve = round((visitors_base / used_months) * 12)
            c_twelve = round((conv_base / used_months) * 12)
            conv_base = c_twelve
            visitors_base = v_twelve
            haircut = 0.13

            # MDE calculation
            cr_base = conv_base / visitors_base if visitors_base > 0 else 0
            cr_min = cr_base # no deviation from base
            cr_max = cr_base * (1 + np.log1p(1.5)) # most positive scenario with log dampening the extreme growth
            mde = 4 * np.sqrt(max((cr_base * (1 - cr_base) / visitors_base), 1e-10))

            # Scaling factors
            scaling_factor_min = 1
            scaling_factor_max = 1 + np.log1p(1.5)

            # Minimum Conversion Rate scaling factor
            mde_min = 4 * np.sqrt(max((cr_min * (1 - cr_min) / visitors_base), 1e-10))
            scaled_mde_min = mde_min * scaling_factor_min

            # Maximum Conversion Rate scaling factor
            mde_max = 4 * np.sqrt(max((cr_max * (1 - cr_max) / visitors_base), 1e-10))
            scaled_mde_max = mde_max * scaling_factor_max

            # Calculate relative MDE for the scaled scenarios
            relative_mde_min = scaled_mde_min / cr_min if cr_min > 0 else 0
            relative_mde_max = scaled_mde_max / cr_max if cr_max > 0 else 0
            relative_mde = mde / cr_base if cr_base > 0 else 0

            adjusted_mde_min = relative_mde_min / (1 + np.log1p(conv_base / visitors_base))
            adjusted_mde_max = relative_mde_max / (1 + np.log1p(conv_base / visitors_base))

            st.write("### Computed statistics")
            #st.write(f"The minimum conversion rate is {cr_min * 100:.2f}")
            #st.write(f"The maximum conversion rate is {cr_max * 100:.2f}")
            st.write(f"The relative MDE is {relative_mde * 100:.2f}%")
            st.write(f"The absolute MDE is {mde:.6f}.")
            st.write(f"The minimum relative MDE is {adjusted_mde_min * 100:.2f}%.")
            st.write(f"The maximum relative MDE is {adjusted_mde_max * 100:.2f}%.")

            # Uplift calculation for range of experiments
        #    def monte_carlo_simulation(
        #        visitors_base,
        #        conv_base,
        #        n_experiments_range,
        #        winrate,
        #        relative_mde_min,
        #        relative_mde_max,
        #        iterations=5000,
        #        small_dataset_mde_scale=10,  # Amplified scaling factor for small datasets
        #        large_dataset_threshold=500_000,  # Threshold for large datasets
        #        gaussian_noise_min_scale=0.0005,  # Noise scale for min CR
        #        gaussian_noise_max_scale=0.001,  # Noise scale for max CR
        #        sigmoid_threshold=19,  # Start diminishing returns after 19 experiments
        #        sigmoid_k=0.05  # Sigmoid slope
        #    ):
        #        def sigmoid(x, x0, k):
        #            return 1 - (1 / (1 + np.exp(-k * (x - x0))))

        #        results = []

        #        for n_experiments in n_experiments_range:
        #            simulated_uplifts_min = []
        #            simulated_uplifts_max = []

                    # Calculate sigmoid multiplier for diminishing returns
        #            sigmoid_multiplier = sigmoid(n_experiments, x0=sigmoid_threshold, k=sigmoid_k)

        #            for _ in range(iterations):
        #                if visitors_base >= large_dataset_threshold:
        #                    random_cr_min = np.clip(
        #                        np.random.normal(loc=conv_base / visitors_base, scale=gaussian_noise_min_scale), 0, 1
        #                    )
        #                    random_cr_max = np.clip(
        #                        np.random.normal(loc=conv_base / visitors_base, scale=gaussian_noise_max_scale), 0, 1
        #                    )
                            # multiply by sigmoid multiplier if appliccable
        #                    uplift_min = sigmoid_multiplier * ((1 + (random_cr_min * (1 - haircut)))**(n_experiments * winrate * (relative_mde_min * 50)) - 1)
        #                    uplift_max = sigmoid_multiplier * ((1 + (random_cr_max * (1 - haircut)))**(n_experiments * winrate * (relative_mde_max * 50)) - 1)
        #                else:
        #                    random_cr_min = np.random.beta(conv_base, max(1, visitors_base - conv_base))
        #                    random_cr_max = np.random.beta(conv_base, max(1, visitors_base - conv_base))
        #                    uplift_min = (1 + (random_cr_min * (1 - haircut)))**(n_experiments * winrate * (relative_mde_min * small_dataset_mde_scale)) - 1
        #                    uplift_max = (1 + (random_cr_max * (1 - haircut)))**(n_experiments * winrate * (relative_mde_max * small_dataset_mde_scale)) - 1

        #                simulated_uplifts_min.append(uplift_min)
        #                simulated_uplifts_max.append(uplift_max)

        #            results.append({
        #                "Experiments": n_experiments,
        #                "Min_Mean_Uplift": round(np.mean(simulated_uplifts_min) * 100, 2),
        #                "Max_Mean_Uplift": round(np.mean(simulated_uplifts_max) * 100, 2),
        #                "Min_Lower_Bound": round(np.percentile(simulated_uplifts_min, 5) * 100, 2),
        #                "Min_Upper_Bound": round(np.percentile(simulated_uplifts_min, 95) * 100, 2),
        #                "Max_Lower_Bound": round(np.percentile(simulated_uplifts_max, 5) * 100, 2),
        #                "Max_Upper_Bound": round(np.percentile(simulated_uplifts_max, 95) * 100, 2),
        #            })

        #        return pd.DataFrame(results)

            def monte_carlo_simulation(
                visitors_base,
                conv_base,
                n_experiments_range,
                winrate,
                adjusted_mde_min,
                adjusted_mde_max,
                iterations = max(5000, min(10000, 50000 // n_experiments_max)),
                small_dataset_mde_scale=10,  # Amplified scaling factor for small datasets
                large_dataset_threshold=500_000,  # Threshold for large datasets
                gaussian_noise_min_scale=0.0005,  # Noise scale for min CR
                gaussian_noise_max_scale=0.001,  # Noise scale for max CR
                sigmoid_threshold=19,  # Start diminishing returns after 19 experiments
                sigmoid_k=0.03  # Sigmoid slope
            ):
                def sigmoid(x, x0, k):
                    return 1 - (1 / (1 + np.exp(-k * (x - x0))))

                results = []

                for n_experiments in n_experiments_range:
                    simulated_uplifts_min = []
                    simulated_uplifts_max = []

                    # Calculate sigmoid multiplier for diminishing returns
                    sigmoid_multiplier = sigmoid(n_experiments, x0=sigmoid_threshold, k=sigmoid_k)

                    for _ in range(iterations):
                        if visitors_base >= large_dataset_threshold:
                            # Apply Gaussian noise for large datasets
                            random_cr_min = np.clip(
                                np.random.normal(loc=conv_base / visitors_base, scale=gaussian_noise_min_scale), 0, 1
                            )
                            random_cr_max = np.clip(
                                np.random.normal(loc=conv_base / visitors_base, scale=gaussian_noise_max_scale), 0, 1
                            )

                            uplift_min = sigmoid_multiplier * (
                                (1 + (random_cr_min * (1 - haircut))) ** (n_experiments * winrate * (adjusted_mde_min * 50)) - 1
                            )
                            uplift_max = sigmoid_multiplier * (
                                (1 + (random_cr_max * (1 - haircut))) ** (n_experiments * winrate * (adjusted_mde_max * 50)) - 1
                            )

                        else:
                            # Apply Beta-distributed noise for small datasets
                            random_cr_min = np.random.beta(conv_base, max(1, visitors_base - conv_base))
                            random_cr_max = np.random.beta(conv_base, max(1, visitors_base - conv_base))

                            uplift_min = (1 + (random_cr_min * (1 - haircut))) ** (
                                n_experiments * winrate * (adjusted_mde_min * small_dataset_mde_scale)
                            ) - 1
                            uplift_max = (1 + (random_cr_max * (1 - haircut))) ** (
                                n_experiments * winrate * (adjusted_mde_max * small_dataset_mde_scale)
                            ) - 1

                        simulated_uplifts_min.append(uplift_min)
                        simulated_uplifts_max.append(uplift_max)

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

        #    def monte_carlo_simulation(
        #        visitors_base,
        #        conv_base,
        #        n_experiments_range,
        #        winrate,
        #        adjusted_mde_min,
        #        adjusted_mde_max,
        #        iterations=5000,
        #        small_dataset_mde_scale=10,
        #        large_dataset_threshold=500_000,
        #        gaussian_noise_min_scale=0.0005,
        #        gaussian_noise_max_scale=0.001,
        #        diminishing_return_rate=0.05, # speed of diminishing returns
        #        diminishing_return_start=19
        #    ):

        #        results = []
        #        n_experiments_total = len(n_experiments_range)  # Total number of experiment sets
        #        progress_bar = st.progress(0)  # Initialize progress bar

        #        for n_experiments_index, n_experiments in enumerate(n_experiments_range):
        #            simulated_uplifts_min = []
        #            simulated_uplifts_max = []

        #            for _ in range(iterations):
        #                cumulative_uplift_min = 0
        #                cumulative_uplift_max = 0
        #                for iteration in range(iterations):
        #                    for i in range(n_experiments):
        #                        diminishing_factor = 1.0  # Default: No diminishing returns

        #                        if i >= diminishing_return_start:
                                    # Use power law or start from 1.
                                    #diminishing_factor = (i - diminishing_return_start + 1)**(-diminishing_return_rate)
        #                            diminishing_factor = 1 / (1 + diminishing_return_rate * (i - diminishing_return_start))

        #                        if visitors_base >= large_dataset_threshold:
        #                            random_cr_min = np.clip(
        #                                np.random.normal(loc=conv_base / visitors_base, scale=gaussian_noise_min_scale), 0, 1
        #                            )
        #                            random_cr_max = np.clip(
        #                                np.random.normal(loc=conv_base / visitors_base, scale=gaussian_noise_max_scale), 0, 1
        #                            )
        #                            uplift_min = diminishing_factor * (
        #                                (1 + (random_cr_min * (1 - haircut))) ** (winrate * (adjusted_mde_min * 50)) - 1
        #                            )
        #                            uplift_max = diminishing_factor * (
        #                                (1 + (random_cr_max * (1 - haircut))) ** (winrate * (adjusted_mde_max * 50)) - 1
        #                            )

        #                        else:
        #                            random_cr_min = np.random.beta(conv_base, max(1, visitors_base - conv_base))
        #                            random_cr_max = np.random.beta(conv_base, max(1, visitors_base - conv_base))

        #                            uplift_min = diminishing_factor * (
        #                                (1 + (random_cr_min * (1 - haircut))) ** (winrate * (adjusted_mde_min * small_dataset_mde_scale)) - 1
        #                            )
        #                            uplift_max = diminishing_factor * (
        #                                (1 + (random_cr_max * (1 - haircut))) ** (winrate * (adjusted_mde_max * small_dataset_mde_scale)) - 1
        #                            )

        #                        cumulative_uplift_min += uplift_min
        #                        cumulative_uplift_max += uplift_max

        #                simulated_uplifts_min.append(cumulative_uplift_min)
        #                simulated_uplifts_max.append(cumulative_uplift_max)

                        # Update Progress Bar
        #                progress_percent = ((n_experiments_index * iterations + iteration + 1) / (n_experiments_total * iterations))
        #                progress_bar.progress(progress_percent)

        #            results.append({
        #                "Experiments": n_experiments,
        #                "Min_Mean_Uplift": round(np.mean(simulated_uplifts_min) * 100, 2),
        #                "Max_Mean_Uplift": round(np.mean(simulated_uplifts_max) * 100, 2),
        #                "Min_Lower_Bound": round(np.percentile(simulated_uplifts_min, 5) * 100, 2),
        #                "Min_Upper_Bound": round(np.percentile(simulated_uplifts_min, 95) * 100, 2),
        #                "Max_Lower_Bound": round(np.percentile(simulated_uplifts_max, 5) * 100, 2),
        #                "Max_Upper_Bound": round(np.percentile(simulated_uplifts_max, 95) * 100, 2),
        #            })

        #        return pd.DataFrame(results)

            # Run simulation with additional parameters
            simulation_df = monte_carlo_simulation(
                visitors_base,
                conv_base,
                n_experiments_range,
                winrate,
                adjusted_mde_min,
                adjusted_mde_max,
                iterations=10000
            )

            filtered_df = simulation_df[['Experiments', 'Min_Mean_Uplift', 'Max_Mean_Uplift']]
            #clean_df = filtered_df.to_string(index=False)
            st.dataframe(
                filtered_df.style.format({"Min_Mean_Uplift": "{:.2f}%", "Max_Mean_Uplift": "{:.2f}%"}),
                use_container_width=False
            )
            #st.text(clean_df)

            # Download simulation results
            csv = simulation_df.to_csv(index=False)
            st.download_button(
                label="Download Simulation Results as CSV",
                data=csv,
                file_name='simulation_results.csv',
                mime='text/csv',
            )
            st.write("")

            # Plot for visualization of estimates
            # Y-Tick Range
            yticks_range = np.arange(
                int(np.floor(min(simulation_df[['Min_Lower_Bound', 'Max_Lower_Bound']].min())) - 1),
                int(np.ceil(max(simulation_df[['Min_Upper_Bound', 'Max_Upper_Bound']].max())) + 1),
                step=1
            )

            # Plotting the Graph
            plt.figure(figsize=(12, 6))
            if len(yticks_range) > 0 and max(yticks_range) != min(yticks_range):
                buffer = max((max(yticks_range) - min(yticks_range)) * 0.05, 1)  # 5% buffer
                plt.ylim(min(yticks_range) - buffer, max(yticks_range) + buffer)
            else:
                plt.ylim(auto=True)  # Auto-scale if no variation

            # Plot Min Uplift and its Confidence Interval
            plt.plot(simulation_df['Experiments'], simulation_df['Min_Mean_Uplift'],
                    label='Min Mean Uplift', color='blue', linewidth=2)
            plt.fill_between(simulation_df['Experiments'],
                            simulation_df['Min_Lower_Bound'],
                            simulation_df['Min_Upper_Bound'],
                            color='blue', alpha=0.2, label='Variability in min. uplift')

            # Plot Max Uplift and its Confidence Interval
            plt.plot(simulation_df['Experiments'], simulation_df['Max_Mean_Uplift'],
                    label='Max Mean Uplift', color='green', linewidth=2)
            plt.fill_between(simulation_df['Experiments'],
                            simulation_df['Max_Lower_Bound'],
                            simulation_df['Max_Upper_Bound'],
                            color='green', alpha=0.2, label='Variability in max. uplift')

            # Add Zero Baseline for Reference
            plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

            # Title and Axis Labels
            plt.title(f"Experiment and Uplift Forecast for {n_experiments_max} Experiments", fontsize=14)
            plt.xlabel("Number of Experiments", fontsize=12)
            plt.ylabel("Uplift (%)", fontsize=12)

            # Set Y-Ticks and Show Grid
            plt.yticks(yticks_range)
            plt.grid(True, linestyle='--', alpha=0.6)

            # Add Legend
            plt.legend(loc='upper left', fontsize=10)

            # Show Plot
            st.pyplot(plt)
            plt.close()
        else:
            st.write("")
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()