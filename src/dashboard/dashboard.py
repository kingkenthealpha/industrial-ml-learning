import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Industrial Predictive Maintenance", layout="wide")

st.title("🏗️ Industrial Predictive Maintenance Dashboard")
st.markdown("---")

# Load Dataset
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(base_dir, "datasets/predictive_maintenance.csv")
    return pd.read_csv(path)

df = load_data()

# Sidebar
st.sidebar.header("Dashboard Settings")
show_raw_data = st.sidebar.checkbox("Show Raw Dataset")

# Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Machines", len(df))
col2.metric("Total Failures", df["Machine failure"].sum())
col3.metric("Failure Rate", f"{(df['Machine failure'].mean()*100):.2f}%")
col4.metric("Avg Torque", f"{df['Torque'].mean():.2f} Nm")

# Failure Analysis
st.subheader("📊 Failure Distribution")
col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots()
    sns.countplot(x="Machine failure", data=df, palette="viridis", ax=ax)
    ax.set_title("Machine Failure Count (0=Normal, 1=Fail)")
    st.pyplot(fig)

with col_b:
    fig, ax = plt.subplots()
    sns.countplot(x="Type", data=df, palette="magma", ax=ax)
    ax.set_title("Machine Distribution by Type")
    st.pyplot(fig)

# Sensor Trends
st.subheader("📈 Sensor Monitoring Trends")
sensor_to_plot = st.selectbox("Select Sensor to Visualize", ["Torque", "Air temperature", "Rotational speed", "Tool wear"])
fig_trend, ax_trend = plt.subplots(figsize=(12, 4))
ax_trend.plot(df[sensor_to_plot].iloc[:1000], color='orange')
ax_trend.set_title(f"{sensor_to_plot} over first 1000 records")
st.pyplot(fig_trend)

# Raw Data Section
if show_raw_data:
    st.subheader("📂 Dataset Preview")
    st.dataframe(df)

st.markdown("---")
st.caption("Powered by Scikit-Learn, FastAPI, and Streamlit | Built for Industrial ML Excellence")
