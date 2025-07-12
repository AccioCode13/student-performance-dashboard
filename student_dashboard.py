# student_dashboard_advanced.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime

# ======================
# CONFIGURATION
# ======================
st.set_page_config(
    page_title="Student Analytics Pro",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# ======================
# DATA LOADING (Replace with your actual data)
# ======================
@st.cache_data
def load_data():
    # Mock data - replace with pd.read_csv() or database connection
    np.random.seed(42)
    data = pd.DataFrame({
        'Student_ID': [f'S{1000+i}' for i in range(200)],
        'Study_Hours': np.random.normal(4.5, 1.5, 200).clip(0, 10),
        'Sleep_Hours': np.random.normal(6.5, 1.2, 200).clip(4, 10),
        'Screen_Time': np.random.normal(4.0, 1.8, 200).clip(0, 12),
        'Attendance': np.random.normal(85, 15, 200).clip(0, 100),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], 200),
        'Performance': np.random.choice(['Low', 'Medium', 'High'], 200, p=[0.2, 0.5, 0.3]),
        'Last_Assessment': pd.to_datetime(['2023-01-15'] + [datetime.now().strftime('%Y-%m-%d') for _ in range(199)])
    })
    return data

df = load_data()

# ======================
# SIDEBAR CONTROLS
# ======================
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=School+Logo", width=150)
    st.header("Dashboard Controls")
    
    # Performance filter
    perf_filter = st.multiselect(
        "Performance Level",
        options=['Low', 'Medium', 'High'],
        default=['High']
    )
    
    # Date range filter
    date_range = st.date_input(
        "Assessment Date Range",
        value=[df['Last_Assessment'].min(), df['Last_Assessment'].max()],
        min_value=df['Last_Assessment'].min(),
        max_value=df['Last_Assessment'].max()
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        show_raw = st.checkbox("Show raw data")
        st.color_picker("Chart color", "#4f8bf9")

# ======================
# MAIN DASHBOARD
# ======================
st.title("📊 Student Performance Analytics Pro")
st.markdown("""
    *Visualizing academic habits and outcomes*  
    Last updated: {}
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")))

# ======================
# METRICS ROW
# ======================
m1, m2, m3, m4 = st.columns(4)
filtered_df = df[df['Performance'].isin(perf_filter)]
m1.metric("Total Students", len(filtered_df))
m2.metric("Avg Study Hours", f"{filtered_df['Study_Hours'].mean():.1f} hrs")
m3.metric("Avg Sleep", f"{filtered_df['Sleep_Hours'].mean():.1f} hrs")
m4.metric("Avg Screen Time", f"{filtered_df['Screen_Time'].mean():.1f} hrs")

# ======================
# VISUALIZATION TABS
# ======================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Distributions", "🧩 Correlations", "🔍 Predictions"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Study vs Sleep")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(
            data=filtered_df,
            x='Study_Hours',
            y='Sleep_Hours',
            hue='Performance',
            palette='viridis',
            ax=ax1
        )
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Attendance Trend")
        fig2, ax2 = plt.subplots()
        sns.lineplot(
            data=filtered_df.sort_values('Last_Assessment'),
            x='Last_Assessment',
            y='Attendance',
            hue='Performance',
            estimator='mean',
            ax=ax2
        )
        plt.xticks(rotation=45)
        st.pyplot(fig2)

with tab2:
    st.subheader("Habit Distributions")
    dist_col = st.selectbox("Select metric", ['Study_Hours', 'Sleep_Hours', 'Screen_Time'])
    fig3, ax3 = plt.subplots()
    sns.boxplot(
        data=filtered_df,
        x='Performance',
        y=dist_col,
        palette='coolwarm'
    )
    st.pyplot(fig3)

with tab3:
    st.subheader("Correlation Matrix")
    num_cols = ['Study_Hours', 'Sleep_Hours', 'Screen_Time', 'Attendance']
    corr_matrix = filtered_df[num_cols].corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

with tab4:
    st.subheader("Performance Predictor")
    st.warning("Note: This uses mock predictions. Connect your real model!")
    
    # Mock prediction interface
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        study = c1.slider("Study Hours", 0.0, 10.0, 5.0)
        sleep = c2.slider("Sleep Hours", 0.0, 12.0, 7.0)
        
        if st.form_submit_button("Predict"):
            # Replace with actual model prediction
            mock_pred = "High" if (study > 4.5 and sleep > 6.5) else ("Medium" if study > 3 else "Low")
            st.success(f"Predicted Performance: **{mock_pred}**")

# ======================
# RAW DATA SECTION
# ======================
if show_raw:
    st.subheader("📋 Raw Data")
    st.dataframe(filtered_df.style.background_gradient(cmap='Blues'), height=300)

# ======================
# FOOTER
# ======================
st.divider()
st.caption("""
    **Analytics Dashboard**  
    Developed by Shreya Sharan
""")

# Run with: streamlit run student_dashboard_advanced.py