import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import welch_anova, qqplot  # Import welch_anova
import streamlit as st

st.set_page_config(
    page_title="Continuous Data Calculator",
    page_icon="🔢",
)

# Preprocess data
def preprocess_data(df):
    errors = []

    # Normalize column names: strip spaces & convert to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Rename columns based on keywords
    for col in df.columns:
        if 'variant' in col:
            df.rename(columns={col: 'experience_variant_label'}, inplace=True)

    # Validate that experience_variant_label exists
    if 'experience_variant_label' not in df.columns:
        errors.append("Column 'experience_variant_label' is missing after preprocessing. Please check your CSV file.")
        return df, errors  # Prevent further processing

    # Check for missing values
    if df['experience_variant_label'].isnull().any():
        errors.append("'experience_variant_label' contains null values.")
    if 'total_item_quantity' in df and df['total_item_quantity'].isnull().any():
        errors.append("'total_item_quantity' contains null values.")
    if 'purchase_revenue' in df and df['purchase_revenue'].isnull().any():
        errors.append("'purchase_revenue' contains null values.")

    # Ensure categorical variable
    df['experience_variant_label'] = pd.Categorical(df['experience_variant_label'])

    # Convert to numeric
    if 'total_item_quantity' in df:
        df['total_item_quantity'] = pd.to_numeric(df['total_item_quantity'], errors='coerce')
    if 'purchase_revenue' in df:
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

    return outliers_mask, model  # Return the initial model

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
    df_copy = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df_copy[kpi] = np.where(df_copy[kpi] < percentile_lower, percentile_lower, df_copy[kpi])
    df_copy[kpi] = np.where(df_copy[kpi] > percentile_upper, percentile_upper, df_copy[kpi])

    return df_copy, lower_bound, upper_bound, percentile_lower, percentile_upper

# Log transform data
def log_transform_data(df, kpi):
    df_copy = df.copy() # Create a copy to avoid SettingWithCopyWarning
    df_copy[kpi] = np.log1p(df_copy[kpi])  # log1p prevents log(0) issues
    return df_copy

# Perform statistical tests and provide conclusions
def perform_stat_tests_and_conclusions(df, kpi, model):
    st.write("## Test Results")
    st.write("Below are the results of your experiment data, dictated by the chosen tests.")
    # Shapiro-Wilk test on RESIDUALS, not the raw data
    shapiro_stat, shapiro_p_val = shapiro(model.resid)
    groups = [group[kpi].dropna() for _, group in df.groupby('experience_variant_label', observed=True)]
    levene_stat, levene_p_val = levene(*groups)

    st.write("### Shapiro-Wilk Test (Normality of Residuals)") # Corrected description
    st.write(f"Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p_val:.4f}")

    st.write("### Levene's Test (Homogeneity of Variance)")
    st.write(f"Statistic = {levene_stat:.4f}, p-value = {levene_p_val:.4f}")

    significant = "no significant"
    if shapiro_p_val >= 0.05 and levene_p_val >= 0.05:
        st.write("### Standard ANOVA")
        st.write("\nPerforming standard ANOVA (data is normal and variances are homogeneous).")
        anova_results = sm.stats.anova_lm(model, typ=2)
        st.write(anova_results)

        if anova_results['PR(>F)'].iloc[0] < 0.05:
            significant = "significant"
            tukey_results = pairwise_tukeyhsd(df[kpi], df['experience_variant_label'])
            st.write("Tukey's Honestly Significant Difference test results:")
            st.write(tukey_results)
    elif levene_p_val >= 0.05:  # Check homogeneity of variance for Welch's ANOVA
        st.write("### Welch's ANOVA")
        st.write("\nPerforming Welch's ANOVA (data is not normal, but variances are homogeneous).")
        # Welch's ANOVA using pingouin
        aov = welch_anova(data=df, dv=kpi, between='experience_variant_label')
        st.write(aov)
        # No post-hoc test is typically used with Welch's ANOVA if only two groups
        if aov['p-unc'][0] < 0.05:
            significant = "significant"

    else:  # Variances are heterogeneous
        if len(df['experience_variant_label'].unique()) > 2:
            st.write("### Kruskal-Wallis Test")
            st.write("\nPerforming Kruskal-Wallis test (non-parametric, variances heterogeneous, three or more groups).")
            statistic, p_value = kruskal(*groups)
            st.write(f"Kruskal-Wallis Test: Statistic = {statistic:.4f}, p-value = {p_value:.4g}")
            if p_value < 0.05:
                significant = "significant"
        else:
            st.write("\nPerforming Mann-Whitney U test (non-parametric, variances heterogeneous, two groups).")
            group1, group2 = groups
            statistic, p_value = mannwhitneyu(group1, group2)
            st.write("### Mann-Whitney U Test")
            st.write(f"Mann-Whitney U Test: Statistic = {statistic:.4f}, p-value = {p_value:.4g}")
            if p_value < 0.05:
                significant = "significant"

    summary_stats = df.groupby('experience_variant_label', observed=True)[kpi].agg(['mean', 'std'])
    highest_mean_variant = summary_stats['mean'].idxmax()
    highest_std_variant = summary_stats['std'].idxmax()
    st.write("### Summary Statistics")
    st.dataframe(summary_stats)

    st.write(f"- The variant with the highest mean is '{highest_mean_variant}' with {summary_stats['mean'].loc[highest_mean_variant]:.2f}.")
    st.write(f"- The variant with the highest standard deviation is '{highest_std_variant}' with {summary_stats['std'].loc[highest_std_variant]:.2f}.")

    st.write("### Conclusion")
    if significant == "significant" and highest_mean_variant == "B" and highest_std_variant == "B":
        st.write(f"Congratulations! Variant B is the <span style='color: green;'>winner</span>!", unsafe_allow_html=True)
    elif significant == "significant" and highest_mean_variant == "A" and highest_std_variant == "A":
        st.write(f"<span style='color: orange;'>Loss prevented</span>! Variant B performed significantly worse.", unsafe_allow_html=True)
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
        outlier_handling = st.selectbox("Select how to handle outliers:", ['None', 'Winsorizing + IQR', 'Log Transform', 'Removal'], help='Choose the method for handling outliers. "None" uses a default > 5 standard deviation definition for detection purposes.')

        method = None
        outlier_stdev = None
        percentile = None

        if outlier_handling not in ['None', 'Log Transform']:
            method = st.selectbox("Select outlier detection method:", ['Standard Deviation', 'Percentile'])
            if method == 'Standard Deviation':
                outlier_stdev = st.selectbox("How many standard deviations define an outlier?", [2, 3, 4, 5])
            elif method == 'Percentile':
                percentile = st.selectbox("Select percentile for Winsorization:", [90, 95, 99])

        outliers_mask, initial_model = detect_outliers(df, kpi, outlier_stdev if method == 'Standard Deviation' else 5)  # Default 5 STD for detection purposes
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
            # --- Outlier Handling ---
            processed_df = df.copy()

            if outlier_handling == 'Winsorizing + IQR':
                processed_df, lower_bound, upper_bound, percentile_lower, percentile_upper = winsorize_iqr_filter(processed_df, kpi, outlier_stdev, percentile)
                if method == 'Percentile':
                    st.write(f"Winsorizing + IQR applied: Capped values between {percentile_lower:.2f} and {percentile_upper:.2f} ({percentile}th percentile-based).")
                elif method == 'Standard Deviation':
                    st.write(f"Winsorizing + IQR applied: Capped values between {percentile_lower:.2f} and {percentile_upper:.2f} (with {outlier_stdev} standard deviations.")
            elif outlier_handling == 'Log Transform':
                processed_df = log_transform_data(processed_df, kpi)
                st.write("Log transformation applied.")
            elif outlier_handling == 'Removal':
                processed_df = processed_df[~outliers_mask]
                st.write(f"Outliers removed: {outliers_mask.sum()} rows affected.")

            # --- Refit the model AFTER outlier handling ---
            model_after = smf.ols(f'{kpi} ~ C(experience_variant_label)', data=processed_df).fit()

            # --- Processed Data Plots (using the processed data) ---
            st.write("### Refitted Data Box Plot")
            sns.boxplot(x='experience_variant_label', y=kpi, data=processed_df)
            st.pyplot(plt)
            plt.clf()

            st.write("### Refitted QQ Plot")
            qqplot(model_after.resid,  marker='o')  # Use residuals from the new model
            plt.title("QQ Plot of Residuals (After Outlier Handling)") # Clear title
            st.pyplot(plt)
            plt.clf()

            st.write("### Refitted Data Histogram with KDE")
            sns.histplot(model_after.resid, kde=True, bins=30) # Use residuals from the new model
            plt.title("Histogram of Residuals (After Outlier Handling) with KDE") # Clear title
            st.pyplot(plt)
            plt.clf()

            perform_stat_tests_and_conclusions(processed_df, kpi, model_after)  # Pass the refitted model

if __name__ == "__main__":
    run()