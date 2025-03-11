import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, levene, shapiro, norm

# Load datasets (cached for performance)
@st.cache_data
def load_data():
    control = pd.read_csv("control_group.csv", delimiter=";")
    test = pd.read_csv("test_group.csv", delimiter=";")
    return control, test

# Data Cleaning Functions
def clean_data(control, test):
    # Add a 'Campaign' column to distinguish groups BEFORE renaming
    control['Campaign_Type'] = 'Control Campaign'
    test['Campaign_Type'] = 'Test Campaign'
    
    # Rename columns for easier access
    control.rename(columns=lambda x: x.replace("# of ", "") if x.startswith("# of ") else x, inplace=True)
    test.rename(columns=lambda x: x.replace("# of ", "") if x.startswith("# of ") else x, inplace=True)

    # Concatenate *after* renaming to avoid duplicate column issues
    full_df = pd.concat([control, test], axis=0, ignore_index=True)

    # Rename "Campaign Name" to "Campaign"
    full_df.rename(columns={"Campaign Name": "Campaign"}, inplace=True)
    full_df.rename(columns={"Campaign_Type": "Campaign Type"}, inplace=True)
    
    # Drop missing values and clean data types
    full_df.dropna(inplace=True)
    full_df["Date"] = pd.to_datetime(full_df["Date"], format="%d.%m.%Y").dt.day  # Use only the day
    numeric_cols = full_df.select_dtypes(include=["float64"]).columns
    full_df[numeric_cols] = full_df[numeric_cols].astype("int")
    
    return full_df

# Helper functions for analysis
def check_normality(col, alpha=0.05):
    stat, p = shapiro(col)
    return "Normal" if p >= alpha else "Not Normal"

def ab_test(control, test, col, alpha=0.05, alt="two-sided"):
    if check_normality(control[col]) == "Normal" and check_normality(test[col]) == "Normal":
        stat, p_lev = levene(control[col], test[col])
        equal_var = p_lev > alpha
        stat, p = ttest_ind(control[col], test[col], equal_var=equal_var, alternative=alt)
    else:
        stat, p = mannwhitneyu(control[col], test[col], alternative=alt)
    conclusion = "Reject H0" if p < alpha else "Fail to Reject H0"
    return p, conclusion

def calculate_mde(control, metric, confidence_level=0.95, power=0.8):
    mean = control[metric].mean()
    std_dev = control[metric].std()
    z_alpha = norm.ppf((1 + confidence_level) / 2)
    z_beta = norm.ppf(power)
    mde = (z_alpha + z_beta) * (std_dev / np.sqrt(len(control)))
    relative_mde = mde / mean
    return mde, relative_mde

# Step 1: Welcome Screen
def welcome_screen():
    st.title("A/B Testing Analytics Tool")
    st.write("Welcome to the A/B Testing Analytics Tool! Follow the guided steps to analyze your data.")
    if st.button("Start Analysis"):
        st.session_state.step = 1

# Step 2: Preview Dataset
def preview_dataset(df):
    st.header("Step 1: Preview Dataset")
    st.subheader("Sample of Combined Dataset")
    st.dataframe(df.sample(5))
    
    if st.button("Next: Compare Statistics"):
        st.session_state.step = 2

# Step 3: Compare Control vs Test Statistics
def compare_statistics(df):
    st.header("Step 2: Compare Control vs Test Statistics")
    
    metric = st.selectbox("Select a metric to compare:", df.select_dtypes(include=["int"]).columns)
    
    control_group = df[df["Campaign Type"] == "Control Campaign"]
    test_group = df[df["Campaign Type"] == "Test Campaign"]
    
    control_mean = control_group[metric].mean()
    test_mean = test_group[metric].mean()
    
    st.write(f"**Control Group {metric} Mean:** {control_mean:.2f}")
    st.write(f"**Test Group {metric} Mean:** {test_mean:.2f}")
    
    diff = test_mean - control_mean
    st.write(f"**Difference:** {diff:.2f}")
    
    # Run statistical tests
    if st.checkbox("Run Statistical Test"):
        p_value, conclusion = ab_test(control_group, test_group, metric)
        st.write(f"P-value: {p_value:.4f}")
        st.write(f"Conclusion: {conclusion}")
    
    if st.button("Next: MDE Calculation"):
        st.session_state.step = 3
    
    if st.button("Back to Preview Dataset"):
        st.session_state.step = 1

# Step 4: MDE Calculation
def mde_calculation(df):
    st.header("Step 3: Minimum Detectable Effect (MDE) Calculation")
    
    metric = st.selectbox("Select a metric for MDE calculation:", df.select_dtypes(include=["int"]).columns)
    
    confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
    power = st.slider("Statistical Power", 0.7, 0.99, 0.8, 0.01)

    control_group = df[df["Campaign Type"] == "Control Campaign"]
    
    mde, relative_mde = calculate_mde(control_group, metric, confidence_level, power)
    
    st.write(f"Minimum Detectable Effect (MDE) for {metric}: {mde:.2f}")
    st.write(f"Relative MDE: {relative_mde:.2%}")
    
    if st.button("Next: Final Analysis"):
        st.session_state.step = 4

    if st.button("Back to Compare Statistics"):
        st.session_state.step = 2

# Step 5: Final Analysis and Results
def final_analysis(df):
    def correlation_analysis(df):
        corr_matrix = df.drop(columns=["Campaign", "Campaign Type", "Day"], errors='ignore').corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        return fig
    
    
    
    # Display final results and insights
    st.header("Step 4: Final Analysis and Results")
    
    # Correlation Heatmap
    st.subheader("Correlation Analysis")
    fig_corr = correlation_analysis(df)
    st.pyplot(fig_corr)
    
    

    if st.button("Back to MDE Calculation"):
        st.session_state.step = 3

# Main App Logic
def main():
    # Initialize session state variable for step tracking
    if 'step' not in st.session_state:
        st.session_state.step = 0

    # Load datasets once at the start and clean them
    control_df_raw, test_df_raw = load_data()
    
    # Data Cleaning and Preparation
    df_cleaned_combined = clean_data(control_df_raw, test_df_raw)

    # Navigation through steps based on session state
    if st.session_state.step == 0:
        welcome_screen()
    elif st.session_state.step == 1:
        preview_dataset(df_cleaned_combined)
    elif st.session_state.step == 2:
        compare_statistics(df_cleaned_combined)
    elif st.session_state.step == 3:
        mde_calculation(df_cleaned_combined)
    elif st.session_state.step == 4:
        final_analysis(df_cleaned_combined)

if __name__ == "__main__":
    main()
