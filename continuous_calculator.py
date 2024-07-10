import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import welch_anova, qqplot
import re
import streamlit as st

# Title and Introduction
st.title("Continuous Data Calculator")
"""
This calculator lets you analyze revenue data or the amount of items of ecommerce transactions (or leads). See the example CSV file for what you need to upload. You're
not limited to just A and B, but can add more labels when applicable (C, D, etc.). Upload your CSV file and select the KPI to analyze. 

The app will identify outliers, fit models, and perform statistical tests.
Based on the test results and the output of the highest average and higest standard deviation, you can determine which variant won.

Happy learning!
"""

# Template CSV download
def get_csv_template():
    data = {
        "experience_variant_label": ["A", "B"],
        "total_item_quantity": [0, 0],
        "purchase_revenue": [0.0, 0.0]
    }
    template_df = pd.DataFrame(data)
    return template_df.to_csv(index=False)

st.download_button(
    label="Download CSV Template",
    data=get_csv_template(),
    file_name="template.csv",
    mime="text/csv",
    help="Click to download a template CSV file for reference."
)

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your CSV file here.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df.columns = [re.sub(r'[^\w]+', '_', col) for col in df.columns]

    st.write("### A random sample of your data:")
    st.write(df.sample(10))

    # Data preparation
    experiment_labels = df['experience_variant_label']
    total_items = df['total_item_quantity']
    transaction_revenue = df['purchase_revenue']

    valid_data = True

    # Check for non-null values
    if experiment_labels.isnull().any():
        st.error("Error: 'experience_variant_label' contains null values.")
        valid_data = False
    if total_items.isnull().any():
        st.error("Error: 'total_item_quantity' contains null values.")
        valid_data = False
    if transaction_revenue.isnull().any():
        st.error("Error: 'purchase_revenue' contains null values.")
        valid_data = False

    # Check for non-negative values
    if (total_items < 0).any():
        st.error("Error: 'total_item_quantity' contains negative values.")
        valid_data = False
    if (transaction_revenue < 0).any():
        st.error("Error: 'purchase_revenue' contains negative values.")
        valid_data = False

    if valid_data:
        unique_labels = df['experience_variant_label'].unique()
        df['experience_variant_label'] = pd.Categorical(df['experience_variant_label'], categories=unique_labels, ordered=True)
        df['total_item_quantity'] = pd.to_numeric(df['total_item_quantity'], errors='coerce')
        df['purchase_revenue'] = pd.to_numeric(df['purchase_revenue'], errors='coerce')

        st.write("Amount of rows: ", len(df))
        st.write("The data types of your cells are:")
        st.write(df.dtypes)

        # Dropdown for KPI selection
        kpi = st.selectbox('Select the KPI to analyze:', ('purchase_revenue', 'total_item_quantity'), help="Select the metric for analysis.")

        st.write(f"### Analyzing {kpi}")

        # Identify outliers
        model = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=df).fit()
        influence = model.get_influence()
        standardized_residuals = influence.resid_studentized_internal
        leverage = influence.hat_matrix_diag
        dffits = influence.dffits[0]

        # Thresholds for identifying potential outliers
        residual_threshold = 2
        leverage_threshold = 2 * (model.df_model + 1) / len(df)
        dffits_threshold = 2 * np.sqrt((model.df_model + 1) / len(df))

        # Identify potential outliers
        outliers = df[(np.abs(standardized_residuals) > residual_threshold) |
                      (leverage > leverage_threshold) |
                      (np.abs(dffits) > dffits_threshold)]

        st.write("### Potential outliers identified:")
        st.write(outliers.head(10))
        st.write("Amount of outliers: ", len(outliers))

        # Remove outliers and return filtered dataframe
        residuals_outliers = np.abs(standardized_residuals) > residual_threshold
        leverage_outliers = leverage > leverage_threshold
        dffits_outliers = np.abs(dffits) > dffits_threshold

        outliers_mask = residuals_outliers | leverage_outliers | dffits_outliers
        non_outliers_mask = ~outliers_mask

        df_filtered = df[non_outliers_mask].copy()
        st.write("### Sample of filtered data (outliers removed):")
        st.write(df_filtered.sample(10))

        # Fit the model without outliers
        model_no_outliers = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=df_filtered).fit()
        st.write("The model has been fitted after outliers have been removed (2 standard deviations).")

        st.write("### QQ Plot of residuals")
        qqplot(model_no_outliers.resid, marker='o')
        st.pyplot(plt)
        plt.clf()

        # Assumption tests
        shapiro_stat, shapiro_p_val = shapiro(model_no_outliers.resid)
        groups = [group[kpi].dropna() for _, group in df_filtered.groupby('experience_variant_label', observed=True)]
        levene_stat, levene_p_val = levene(*groups)

        summary_stats = df_filtered.groupby('experience_variant_label', observed=True)[kpi].agg(['mean', 'std'])
        highest_mean_variant = summary_stats['mean'].idxmax()
        highest_mean = summary_stats.loc[highest_mean_variant, 'mean']
        highest_std_variant = summary_stats['std'].idxmax()
        highest_std = summary_stats.loc[highest_std_variant, 'std']

        st.write("### Box plot")
        st.write(summary_stats)
        sns.boxplot(x='experience_variant_label', y=kpi, data=df_filtered)
        st.pyplot(plt)
        plt.clf()

        st.write("Significance threshold = 95%")

        if shapiro_p_val < 0.05 or levene_p_val < 0.05:
            if levene_p_val < 0.05:
                st.write("\nSince the variance is not homogeneous, Welch's ANOVA was performed.")
                
                welch_results = welch_anova(dv=kpi, between='experience_variant_label', data=df_filtered)
                st.write(welch_results)
                
                significant = 'significant' if welch_results['p-unc'].iloc[0] < 0.05 else 'not significant'
                st.write(f"\nConclusion: Welch's ANOVA was performed due to non-homogeneous variance, " +
                      f"with results suggesting {significant} differences between the groups with a " + 
                      f"p-value of {welch_results['p-unc'].iloc[0]:.4f}.")
            
            else:
                unique_variants = df_filtered['experience_variant_label'].nunique()

                if unique_variants > 2:
                    st.write("\nWe used the non-parametric Kruskal-Wallis test due to non-normal distribution " +
                          f"between more than 2 groups.")
                    
                    statistic, p_value = kruskal(*groups)
                    st.write(f"Kruskal-Wallis H-test statistic: {statistic:.4f}, P-value: {p_value:.4g}")
                    
                    significant = 'significant' if p_value < 0.05 else 'not significant'
                    st.write(f"\nConclusion: The Kruskal-Wallis test results " +
                          f"suggest {significant} differences between the groups.")
                elif unique_variants == 2:
                    st.write("\nWe used the non-parametric Mann-Whitney U test for two groups.")
                    
                    group1, group2 = [df_filtered[df_filtered['experience_variant_label'] == variant][kpi] 
                                      for variant in df_filtered['experience_variant_label'].unique()]
                    statistic, p_value = mannwhitneyu(group1, group2)
                    
                    st.write(f"Mann-Whitney U test statistic: {statistic:.4f}, P-value: {p_value:.4g}")
                    
                    significant = 'significant' if p_value < 0.05 else 'not significant'
                    st.write(f"\nConclusion: The Mann-Whitney U test results " +
                          f"suggest {significant} differences between the two groups.")

        else:
            st.write("\nThe data shows a normal distribution with homogeneous variance, so a standard ANOVA was performed.")
            anova_results = sm.stats.anova_lm(model_no_outliers, typ=2)
            
            st.write(anova_results)
            significant = 'significant' if anova_results['PR(>F)'].iloc[0] < 0.05 else 'not significant'
            
            st.write(f"\nConclusion: ANOVA was performed because the data shows a normal distribution with homogeneous variance, " +
                  f"with results suggesting {significant} differences between the groups.")

            if anova_results['PR(>F)'][0] < 0.05:
                tukey_results = pairwise_tukeyhsd(endog=df_filtered[kpi], groups=df_filtered['experience_variant_label'], 
                                                  alpha=0.05)
                
                st.write("\nTukey's HSD test results for multiple comparisons:")
                st.write(tukey_results)
                
                tukey_results.plot_simultaneous()
                plt.show()
                plt.clf()

        st.write(f"\nThe variant with the highest mean in '{kpi}' is the {highest_mean_variant} group " +
              f"with an average of {highest_mean:.2f}.")
        st.write(f"The variant with the highest spread of values in '{kpi}' " +
              f"is the {highest_std_variant} group with a standard deviation of {highest_std:.2f}.")
else:
    st.info("Please upload a CSV file to proceed.")