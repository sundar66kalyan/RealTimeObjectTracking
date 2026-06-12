import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚦 Intelligent Traffic Analytics Dashboard")

# Current report
df = pd.read_csv(
    "outputs/vehicle_report.csv"
)

chart_df = df[
    df["Vehicle Type"] != "Total"
]

# Historical data
history_df = pd.read_csv(
    "outputs/history/traffic_history.csv"
)

# KPIs
total_vehicles = history_df.iloc[-1]["Total"]

most_common_vehicle = (
    chart_df.sort_values(
        by="Count",
        ascending=False
    )
    .iloc[0]["Vehicle Type"]
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Vehicles",
        int(total_vehicles)
    )

with col2:
    st.metric(
        "Most Common Vehicle",
        most_common_vehicle
    )

with col3:

    if total_vehicles > 20:
        traffic_level = "High"
    elif total_vehicles > 10:
        traffic_level = "Medium"
    else:
        traffic_level = "Low"

    st.metric(
        "Traffic Density",
        traffic_level
    )

# Vehicle Table
st.subheader(
    "Vehicle Summary"
)

st.dataframe(
    chart_df,
    use_container_width=True
)

# Bar Chart
st.subheader(
    "Vehicle Distribution"
)

bar_fig = px.bar(
    chart_df,
    x="Vehicle Type",
    y="Count",
    text="Count"
)

st.plotly_chart(
    bar_fig,
    use_container_width=True
)

# Pie Chart
st.subheader(
    "Vehicle Share"
)

pie_fig = px.pie(
    chart_df,
    names="Vehicle Type",
    values="Count"
)

st.plotly_chart(
    pie_fig,
    use_container_width=True
)

# Historical Trend
st.subheader(
    "Historical Traffic Trend"
)

history_df["Timestamp"] = pd.to_datetime(
    history_df["Timestamp"]
)

line_fig = px.line(
    history_df,
    x="Timestamp",
    y="Total",
    markers=True,
    title="Traffic Over Time"
)

st.plotly_chart(
    line_fig,
    use_container_width=True
)

# Insights
st.subheader(
    "AI Traffic Insights"
)

st.success(
    f"Most common vehicle detected: {most_common_vehicle}"
)

st.info(
    f"Traffic density currently: {traffic_level}"
)

st.warning(
    f"Historical records stored: {len(history_df)}"
)

# Download
csv = df.to_csv(index=False)

st.download_button(
    label="📥 Download CSV Report",
    data=csv,
    file_name="vehicle_report.csv",
    mime="text/csv"
)


st.subheader("🚨 Traffic Alerts")

alert_df = pd.read_csv(
    "outputs/traffic_alerts.csv"
)

st.dataframe(
    alert_df.tail(10),
    use_container_width=True
)