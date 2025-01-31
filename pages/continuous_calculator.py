import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import welch_anova, qqplot
import streamlit as st

st.set_page_config(
    page_title="Continuous Data Calculator",
    page_icon="🔢",
)

# Preprocess data
def preprocess_data(df):
    for col in df.columns:
        if 'purchase' in col:
            df.rename(columns={col: 'purchase_revenue'}, inplace=True)
        elif 'total' in col:
            df.rename(columns={col: 'total_item_quantity'}, inplace=True)
        elif 'variant' in col:
            df.rename(columns={col: 'experience_variant_label'}, inplace=True)

    errors = []
    if df['experience_variant_label'].isnull().any():
        errors.append("'experience_variant_label' contains null values.")
    if df['total_item_quantity'].isnull().any():
        errors.append("'total_item_quantity' contains null values.")
    if df['purchase_revenue'].isnull().any():
        errors.append("'purchase_revenue' contains null values.")
    if (df['total_item_quantity'] < 0).any():
        errors.append("'total_item_quantity' contains negative values.")
    if (df['purchase_revenue'] < 0).any():
        errors.append("'purchase_revenue' contains negative values.")

    unique_labels = df['experience_variant_label'].unique()
    df['experience_variant_label'] = pd.Categorical(df['experience_variant_label'], categories=unique_labels, ordered=True)
    df['total_item_quantity'] = pd.to_numeric(df['total_item_quantity'], errors='coerce')
    df['purchase_revenue'] = pd.to_numeric(df['purchase_revenue'], errors='coerce')

    return df, errors

# Detect outliers
def detect_outliers(df, kpi, outlier_stdev):
    model = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=df).fit()
    influence = model.get_influence()
    standardized_residuals = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    dffits = influence.dffits[0]

    residual_threshold = outlier_stdev
    leverage_threshold = outlier_stdev * (model.df_model + 1) / len(df)
    dffits_threshold = outlier_stdev * np.sqrt((model.df_model + 1) / len(df))

    residuals_outliers = np.abs(standardized_residuals) > residual_threshold
    leverage_outliers = leverage > leverage_threshold
    dffits_outliers = np.abs(dffits) > dffits_threshold
    outliers_mask = residuals_outliers | leverage_outliers | dffits_outliers

    return outliers_mask, model

# Winsorize and IQR filter combined
def winsorize_iqr_filter(df, kpi, outlier_stdev, percentile):
    # Ensure outlier_stdev is not None before using it
    if outlier_stdev is None:
        outlier_stdev = 3  # Default value if not provided
    
    # Calculate IQR
    Q1 = df[kpi].quantile(0.25)
    Q3 = df[kpi].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define upper and lower bounds using both IQR and standard deviation
    lower_bound = max(Q1 - 1.5 * IQR, df[kpi].mean() - (outlier_stdev * df[kpi].std()))
    upper_bound = min(Q3 + 1.5 * IQR, df[kpi].mean() + (outlier_stdev * df[kpi].std()))
    
    # Ensure percentile is not None before using it
    if percentile is None:
        percentile = 95  # Default percentile if not provided
    
    # Alternative Winsorization using percentile-based capping
    lower_percentile = (100 - percentile) / 200
    upper_percentile = 1 - lower_percentile
    percentile_lower = df[kpi].quantile(lower_percentile)
    percentile_upper = df[kpi].quantile(upper_percentile)
    
    # Apply Winsorization
    df[kpi] = np.where(df[kpi] < percentile_lower, percentile_lower, df[kpi])
    df[kpi] = np.where(df[kpi] > percentile_upper, percentile_upper, df[kpi])
    
    return df, lower_bound, upper_bound, percentile_lower, percentile_upper

# Log transform data
def log_transform_data(df, kpi):
    df[kpi] = np.log1p(df[kpi])  # log1p prevents log(0) issues
    return df

# Perform statistical tests and provide conclusions
def perform_stat_tests_and_conclusions(df, kpi, model):
    st.write("### Test Results")
    shapiro_stat, shapiro_p_val = shapiro(model.resid)
    groups = [group[kpi].dropna() for _, group in df.groupby('experience_variant_label', observed=True)]
    levene_stat, levene_p_val = levene(*groups)

    st.write("### Shapiro-Wilk Test (Normality)")
    st.write(f"Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p_val:.4f}")

    st.write("### Levene's Test (Homogeneity of Variance)")
    st.write(f"Statistic = {levene_stat:.4f}, p-value = {levene_p_val:.4f}")

    significant = "no significant"
    if shapiro_p_val >= 0.05 and levene_p_val >= 0.05:
        st.write("\nPerforming standard ANOVA (data is normal and variances are homogeneous).")
        anova_results = sm.stats.anova_lm(model, typ=2)
        st.write(anova_results)

        if anova_results['PR(>F)'].iloc[0] < 0.05:
            significant = "significant"
            tukey_results = pairwise_tukeyhsd(df[kpi], df['experience_variant_label'])
            st.write("Tukey's Honestly Significant Difference test results:")
            st.write(tukey_results)
    else:
        if len(df['experience_variant_label'].unique()) > 2:
            st.write("\nPerforming Kruskal-Wallis test.")
            statistic, p_value = kruskal(*groups)
            st.write(f"Kruskal-Wallis Test: Statistic = {statistic:.4f}, p-value = {p_value:.4g}")
            if p_value < 0.05:
                significant = "significant"
        else:
            st.write("\nPerforming Mann-Whitney U test.")
            group1, group2 = groups
            statistic, p_value = mannwhitneyu(group1, group2)
            st.write(f"Mann-Whitney U Test: Statistic = {statistic:.4f}, p-value = {p_value:.4g}")
            if p_value < 0.05:
                significant = "significant"

    summary_stats = df.groupby('experience_variant_label', observed=True)[kpi].agg(['mean', 'std'])
    highest_mean_variant = summary_stats['mean'].idxmax()
    highest_std_variant = summary_stats['std'].idxmax()
    st.write("### Summary Statistics")
    st.dataframe(summary_stats)

    st.write(f"The variant with the highest mean is '{highest_mean_variant}' with {summary_stats['mean'].loc[highest_mean_variant]:.2f}.")
    st.write(f"The variant with the highest standard deviation is '{highest_std_variant}' with {summary_stats['std'].loc[highest_std_variant]:.2f}.")

    if significant == "significant" and highest_mean_variant == "B" and highest_std_variant == "B":
        st.write(f"Congratulations! Variant B is the <span style='color: green;'>winner</span>!", unsafe_allow_html=True)
    elif significant == "significant" and highest_mean_variant == "A" and highest_std_variant == "A":
        st.write(f"<span style='color: orange;'>Loss prevented</span>! Variant A performed significantly worse.", unsafe_allow_html=True)
    else:
        st.write("No significant differences detected. More data may be needed.")

# Main Streamlit app
def run():
    st.title("Continuous Data Calculator")
    """
    This calculator lets you analyze revenue data or the amount of items of ecommerce transactions (or leads) for your online experiments. See the example CSV file for what you need to upload. 
    You're not limited to just A and B, but can add more labels when applicable (C, D, etc.).

    The app will identify outliers, fit models, and perform statistical tests. Based on the test results and the output of the highest average and highest standard deviation, you can determine which variant won.

    How to use:
    1. Upload the CSV (download the example to see the column names)
    2. Select the KPI to analyze
    3. Select how to handle outliers (Winsorization, log transform or removal)
    4. Choose outlier handling method (percentile, standard deviation)
    5. Push the button!

    When choosing an outlier handling method:
    - Choose Winsorization to cap outlier values at a chosen threshold and not lose data points.
    - Choose log transform when the data is heavily right-skewed to compress high values.
    - Choose removal when there are very few, very extreme values that affect conclusions.

    """
    st.download_button(
        label="Download CSV Template",
        data=pd.DataFrame({
            "experience_variant_label": ["A", "B", "B", "A"],
            "total_item_quantity": [5, 2, 4, 1],
            "purchase_revenue": [114.35, 45.74, 91.48, 22.87]
        }).to_csv(index=False),
        file_name="template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df, errors = preprocess_data(df)
        if errors:
            for error in errors:
                st.error(error)
            return

        st.write("### A random sample of your data:")
        st.write(df.sample(10))

        kpi = st.selectbox("Select the KPI to analyze:", ['purchase_revenue', 'total_item_quantity'])
        outlier_handling = st.selectbox("Select how to handle outliers:", ['None', 'Winsorizing + IQR', 'Log Transform', 'Removal'], help='Choose the method for handling outliers.')
        
        method = None
        outlier_stdev = None
        percentile = None
        
        if outlier_handling not in ['None', 'Log Transform']:
            method = st.selectbox("Select outlier detection method:", ['Standard Deviation', 'Percentile'])
            if method == 'Standard Deviation':
                outlier_stdev = st.selectbox("How many standard deviations define an outlier?", [2, 3, 4, 5])
            elif method == 'Percentile':
                percentile = st.selectbox("Select percentile for Winsorization:", [90, 95, 99])
        
        outliers_mask, model = detect_outliers(df, kpi, outlier_stdev if method == 'Standard Deviation' else 3)  # Default 3 STD for detection purposes
        st.write(f"Number of detected outliers: {outliers_mask.sum()}")

        # Show raw data plots before any processing
        st.write("### Raw Data Box Plot")
        sns.boxplot(x='experience_variant_label', y=kpi, data=df)
        st.pyplot(plt)
        plt.clf()
        
        st.write("### Raw Data Histogram with KDE")
        sns.histplot(df[kpi], kde=True, bins=30)
        plt.title("Raw Data Histogram with KDE")
        st.pyplot(plt)
        plt.clf()

        if st.button("Calculate my test results"):
            if outlier_handling == 'Winsorizing + IQR':
                df, lower_bound, upper_bound, percentile_lower, percentile_upper = winsorize_iqr_filter(df, kpi, outlier_stdev, percentile)
                st.write(f"Winsorizing + IQR applied: Capped values between {percentile_lower:.2f} and {percentile_upper:.2f} ({percentile}th percentile-based).")
            elif outlier_handling == 'Log Transform':
                df = log_transform_data(df, kpi)
                st.write("Log transformation applied.")
            elif outlier_handling == 'Removal':
                df = df[~outliers_mask]
                st.write(f"Outliers removed: {outliers_mask.sum()} rows affected.")

            st.write("### Processed Data Box Plot")
            sns.boxplot(x='experience_variant_label', y=kpi, data=df)
            st.pyplot(plt)
            plt.clf()

            st.write("### QQ Plot")
            qqplot(model.resid, marker='o')
            st.pyplot(plt)
            plt.clf()

            st.write("### Processed Data Histogram with KDE")
            sns.histplot(model.resid, kde=True, bins=30)
            plt.title("Processed Data Histogram with KDE")
            st.pyplot(plt)
            plt.clf()

            perform_stat_tests_and_conclusions(df, kpi, model)

if __name__ == "__main__":
    run()
