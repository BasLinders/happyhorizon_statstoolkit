import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st

def run():
    st.set_page_config(
        page_title="Interaction Effect calculator",
        page_icon="🔢",
    )

    # Initialize session state defaults
    st.session_state.setdefault("AA_u", 0)
    st.session_state.setdefault("AB_u", 0)
    st.session_state.setdefault("BA_u", 0)
    st.session_state.setdefault("BB_u", 0)
    st.session_state.setdefault("AA_c", 0)
    st.session_state.setdefault("AB_c", 0)
    st.session_state.setdefault("BA_c", 0)
    st.session_state.setdefault("BB_c", 0)

    st.title("Interaction Effect Calculator")
    """
    This calculator lets you see if your variants from two experiments that ran concurrently influenced eachother on the KPI that
    you're measuring. The most important thing is that you fetch data that is accumulated in the combinations AA, AB, BA, BB correctly.

    Enter that data in the calculator below and the algorithm will determine whether or not to interpret the experiments with caution.

    Happy learning!
    """

    # Define the data input fields

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Visitors")
        st.session_state.AA_u = st.number_input("AA Visitors", value=st.session_state.AA_u)
        AA_u = st.session_state.AA_u
        st.session_state.AB_u = st.number_input("AB Visitors", value=st.session_state.AB_u)
        AB_u = st.session_state.AB_u
        st.session_state.BA_u = st.number_input("BA Visitors", value=st.session_state.BA_u)
        BA_u = st.session_state.BA_u
        st.session_state.BB_u = st.number_input("BB Visitors", value=st.session_state.BB_u)
        BB_u = st.session_state.BB_u

    with col2:
        st.write("### Conversions")
        st.session_state.AA_c = st.number_input("AA Conversions", value=st.session_state.AA_c)
        AA_c = st.session_state.AA_c
        st.session_state.AB_c = st.number_input("AB Conversions", value=st.session_state.AB_c)
        AB_c = st.session_state.AB_c
        st.session_state.BA_c = st.number_input("BA Conversions", value=st.session_state.BA_c)
        BA_c = st.session_state.BA_c
        st.session_state.BB_c = st.number_input("BB Conversions", value=st.session_state.BB_c)
        BB_c = st.session_state.BB_c

    st.write("")
    if st.button("Calculate interaction effect"):
        if AA_u > 0 and AB_u > 0 and BA_u > 0 and BB_u > 0 and AA_c > 0 and AB_c > 0 and BA_c > 0 and BB_c > 0:
            # Creating dataframes for each combination
            data_AA = pd.DataFrame({'Combination': 'AA', 'User': range(1, AA_u+1), 'Conversion': [1]*AA_c + [0]*(AA_u - AA_c)})
            data_AB = pd.DataFrame({'Combination': 'AB', 'User': range(1, AB_u+1), 'Conversion': [1]*AB_c + [0]*(AB_u - AB_c)})
            data_BA = pd.DataFrame({'Combination': 'BA', 'User': range(1, BA_u+1), 'Conversion': [1]*BA_c + [0]*(BA_u - BA_c)})
            data_BB = pd.DataFrame({'Combination': 'BB', 'User': range(1, BB_u+1), 'Conversion': [1]*BB_c + [0]*(BB_u - BB_c)})

            # Combine all dataframes into a single frame
            data_long = pd.concat([data_AA, data_AB, data_BA, data_BB])

            # Ensure 'Combination' is treated as a categorical variable before creating dummies
            data_long['Combination'] = data_long['Combination'].astype('category')

            # Create dummy variables for the combinations
            data_long = pd.get_dummies(data_long, columns=['Combination'], drop_first=True)

            # Convert boolean columns to integers
            for column in data_long.columns:
                if data_long[column].dtype == bool:
                    data_long[column] = data_long[column].astype(int)

            # Logistic regression
            X = data_long.iloc[:, 2:]  # independent variables
            y = data_long['Conversion']  # dependent variable
            X = sm.add_constant(X)  # adding a constant

            # Fit the model
            try:
                model = sm.Logit(y, X).fit()
                st.write(model.summary())

                # Extracting coefficients, standard errors, z-scores, p-values, and confidence intervals
                coefficients_table = model.summary2().tables[1]
                #st.write("\nCoefficients:\n", coefficients_table)

                # Calculate and print additional values
                def mc_fadden_pseudo_r2(model):
                    return 1 - model.llf / model.llnull

                mc_fadden_r2 = mc_fadden_pseudo_r2(model)
                #st.write("")
                #st.write(f"Pseudo R-squared: {mc_fadden_r2}")
                #st.write(f"Log-Likelihood: {model.llf}")
                #st.write(f"Log-Likelihood (Null Model): {model.llnull}")

                # Analyzing the interaction effect
                #st.write("")
                #st.write("\n# Insights from the logistic regression model:")
                #st.write("The coefficients of the model suggest these influences of the combinations on conversion rates:")
                #for combination in ['Combination_AB', 'Combination_BA', 'Combination_BB']:
                #    coef = coefficients_table.loc[combination, 'Coef.']
                #    p_value = coefficients_table.loc[combination, 'P>|z|']
                #    st.write(f"- {combination}: Coef = {coef:.4f}, P-value = {p_value:.2e}")
                #    if p_value < 0.05:
                #        st.write(f" This combination has a significant impact on the likelihood of conversion, with a coefficient of {coef:.4f}.")

                # Extracting p-values for each combination
                p_values = coefficients_table.loc[['Combination_AB', 'Combination_BA', 'Combination_BB'], 'P>|z|']

                # Identifying the combination with the lowest p-value
                best_combination = p_values.idxmin()
                best_p_value = p_values.min()
                worst_combination = p_values.idxmax()
                worst_p_value = p_values.max()

                #st.write("")
                #st.write(f"\nThe best performing combination based on p-value is {best_combination} (P-value = {best_p_value:.2e}).")
                #st.write(f"The worst performing combination based on p-value is {worst_combination} (P-value = {worst_p_value:.2e}).")

                # Extract scalar value for the p-value of Combination_BB
                bb_p_value = coefficients_table.loc['Combination_BB', 'P>|z|'] 
                st.write("### Results summary")
                st.write("Below is an interpretation of the coefficients and p-values in the model. The focus lies on the interaction between both 'B' variants. " \
                         "If there is no measurable negative interaction in that group or other groups, you're safe to rely on individual test results for inference.")
                if bb_p_value < .05:
                    st.write("")
                    st.write(f"\nVisitors that saw both your test variants converted significantly worse at p-value {bb_p_value:.2e}. " \
                            f"Interpret your experiment results with care.")
                else:
                    for combination in ['Combination_AB', 'Combination_BA', 'Combination_BB']:
                        coef = coefficients_table.loc[combination, 'Coef.']
                        p_value = coefficients_table.loc[combination, 'P>|z|']

                        if p_value < 0.05 and coef < 0:
                            st.write(f"Combination {combination} has a significant impact on the likelihood of conversion, with a p-value of {p_value:.2e} "
                                     f"and a negative coefficient of {coef:.4f}.")
                            st.write("")
                            st.write("\nVisitors who interacted with both your test variants (group BB) didn't react significantly more negatively than other visitors, " \
                                    "but the impact on this group indicates a possibly negative impact when implenting this change and warrants caution.")
                            st.write("")
                        else:
                            st.write("")
                            st.write("\nVisitors who interacted with both your test variants (group BB) didn't react significantly more negatively than other visitors; "\
                                     "you can interpet the results of your experiments as you normally would.")
                            st.write("")

            except Exception as e:
                st.write(f"Error fitting the model: {e}")

            # Visualization
            predict_data = pd.DataFrame({
                'const': 1,
                'Combination_AB': [0, 1, 0, 0],
                'Combination_BA': [0, 0, 1, 0],
                'Combination_BB': [0, 0, 0, 1]
            })

            # Predict probabilities
            predict_data['prob'] = model.predict(predict_data)

            # Function to format y-axis ticks
            def y_fmt(x, _):
                return f'{x:.3f}'

            # Plotting the interaction effects
            fig, ax = plt.subplots()

            # Plot each line
            ax.plot(['A', 'B'], [predict_data['prob'][0], predict_data['prob'][1]], label='Test1 A', marker='o', color='blue')
            ax.plot(['A', 'B'], [predict_data['prob'][2], predict_data['prob'][3]], label='Test1 B', marker='o', color='orange')

            # Adding text annotations
            ax.text('A', predict_data['prob'][0], 'AA', horizontalalignment='right', color='blue')
            ax.text('B', predict_data['prob'][1], 'AB', horizontalalignment='left', color='blue')
            ax.text('A', predict_data['prob'][2], 'BA', horizontalalignment='right', color='orange')
            ax.text('B', predict_data['prob'][3], 'BB', horizontalalignment='left', color='orange')

            # Setting plot labels and title
            ax.set_xlabel('Test 2 Level')
            ax.set_ylabel('Probability of Conversion')
            ax.set_title('Interaction Effect of Test1 and Test2 on Conversion')
            ax.legend()
            ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
            ax.grid(True)

            # Display the plot in Streamlit
            st.pyplot(fig)
        else:
            st.write("")
            st.write("<span style='color: #ff6600;'>*Please enter valid inputs for all fields</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()