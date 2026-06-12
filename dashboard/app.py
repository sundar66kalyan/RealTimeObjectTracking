```python
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚦 Intelligent Traffic Analytics Dashboard")

# -------------------------
# Vehicle Report
# -------------------------

report_file = "outputs/vehicle_report.csv"

if not os.path.exists(report_file):
    st.warning("Vehicle report not found.")

    uploaded_report = st.file_uploader(
        "Upload vehicle_report.csv",
        type=["csv"]
    )

    if uploaded_report is not None:
        df = pd.read_csv(uploaded_report)
    else:
        st.stop()
else:
    df = pd.read_csv(report_file)

chart_df = df[df["Vehicle Type"] != "Total"]

# -------------------------
# Historical Data
# -------------------------

history_file = "outputs/history/traffic_history.csv"

if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
else:
    history_df = pd.DataFrame(
        {
            "Timestamp": [],
            "Total": []
        }
    )

# -------------------------
# KPIs
# -------------------------

if len(history_df) > 0:

    total_vehicles = history_df.iloc[-1]["Total"]

else:

    total_vehicles = chart_df["Count"].sum()

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

# -------------------------
# Vehicle Table
# -------------------------

st.subheader("Vehicle Summary")

st.dataframe(
    chart_df,
    use_container_width=True
)

# -------------------------
# Bar Chart
# -------------------------

st.subheader("Vehicle Distribution")

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

# -------------------------
# Pie Chart
# -------------------------

st.subheader("Vehicle Share")

pie_fig = px.pie(
    chart_df,
    names="Vehicle Type",
    values="Count"
)

st.plotly_chart(
    pie_fig,
    use_container_width=True
)

# -------------------------
# Historical Trend
# -------------------------

st.subheader("Historical Traffic Trend")

if len(history_df) > 0:

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

else:

    st.info(
        "No historical analytics data available."
    )

# -------------------------
# Insights
# -------------------------

st.subheader("AI Traffic Insights")

st.success(
    f"Most common vehicle detected: {most_common_vehicle}"
)

st.info(
    f"Traffic density currently: {traffic_level}"
)

if len(history_df) > 0:
    st.warning(
        f"Historical records stored: {len(history_df)}"
    )

# -------------------------
# CSV Download
# -------------------------

csv = df.to_csv(index=False)

st.download_button(
    label="📥 Download CSV Report",
    data=csv,
    file_name="vehicle_report.csv",
    mime="text/csv"
)

# -------------------------
# Traffic Alerts
# -------------------------

st.subheader("🚨 Traffic Alerts")

alert_file = "outputs/traffic_alerts.csv"

if os.path.exists(alert_file):

    alert_df = pd.read_csv(alert_file)

    st.dataframe(
        alert_df.tail(10),
        use_container_width=True
    )

else:

    st.info(
        "No traffic alerts available."
    )
```
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚦 Intelligent Traffic Analytics Dashboard")

# -------------------------
# Vehicle Report
# -------------------------

report_file = "outputs/vehicle_report.csv"

if not os.path.exists(report_file):
    st.warning("Vehicle report not found.")

    uploaded_report = st.file_uploader(
        "Upload vehicle_report.csv",
        type=["csv"]
    )

    if uploaded_report is not None:
        df = pd.read_csv(uploaded_report)
    else:
        st.stop()
else:
    df = pd.read_csv(report_file)

chart_df = df[df["Vehicle Type"] != "Total"]

# -------------------------
# Historical Data
# -------------------------

history_file = "outputs/history/traffic_history.csv"

if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
else:
    history_df = pd.DataFrame(
        {
            "Timestamp": [],
            "Total": []
        }
    )

# -------------------------
# KPIs
# -------------------------

if len(history_df) > 0:

    total_vehicles = history_df.iloc[-1]["Total"]

else:

    total_vehicles = chart_df["Count"].sum()

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

# -------------------------
# Vehicle Table
# -------------------------

st.subheader("Vehicle Summary")

st.dataframe(
    chart_df,
    use_container_width=True
)

# -------------------------
# Bar Chart
# -------------------------

st.subheader("Vehicle Distribution")

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

# -------------------------
# Pie Chart
# -------------------------

st.subheader("Vehicle Share")

pie_fig = px.pie(
    chart_df,
    names="Vehicle Type",
    values="Count"
)

st.plotly_chart(
    pie_fig,
    use_container_width=True
)

# -------------------------
# Historical Trend
# -------------------------

st.subheader("Historical Traffic Trend")

if len(history_df) > 0:

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

else:

    st.info(
        "No historical analytics data available."
    )

# -------------------------
# Insights
# -------------------------

st.subheader("AI Traffic Insights")

st.success(
    f"Most common vehicle detected: {most_common_vehicle}"
)

st.info(
    f"Traffic density currently: {traffic_level}"
)

if len(history_df) > 0:
    st.warning(
        f"Historical records stored: {len(history_df)}"
    )

# -------------------------
# CSV Download
# -------------------------

csv = df.to_csv(index=False)

st.download_button(
    label="📥 Download CSV Report",
    data=csv,
    file_name="vehicle_report.csv",
    mime="text/csv"
)

# -------------------------
# Traffic Alerts
# -------------------------

st.subheader("🚨 Traffic Alerts")

alert_file = "outputs/traffic_alerts.csv"

if os.path.exists(alert_file):

    alert_df = pd.read_csv(alert_file)

    st.dataframe(
        alert_df.tail(10),
        use_container_width=True
    )

else:

    st.info(
        "No traffic alerts available."
    )
```
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚦 Intelligent Traffic Analytics Dashboard")

# -------------------------
# Vehicle Report
# -------------------------

report_file = "outputs/vehicle_report.csv"

if not os.path.exists(report_file):
    st.warning("Vehicle report not found.")

    uploaded_report = st.file_uploader(
        "Upload vehicle_report.csv",
        type=["csv"]
    )

    if uploaded_report is not None:
        df = pd.read_csv(uploaded_report)
    else:
        st.stop()
else:
    df = pd.read_csv(report_file)

chart_df = df[df["Vehicle Type"] != "Total"]

# -------------------------
# Historical Data
# -------------------------

history_file = "outputs/history/traffic_history.csv"

if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
else:
    history_df = pd.DataFrame(
        {
            "Timestamp": [],
            "Total": []
        }
    )

# -------------------------
# KPIs
# -------------------------

if len(history_df) > 0:

    total_vehicles = history_df.iloc[-1]["Total"]

else:

    total_vehicles = chart_df["Count"].sum()

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

# -------------------------
# Vehicle Table
# -------------------------

st.subheader("Vehicle Summary")

st.dataframe(
    chart_df,
    use_container_width=True
)

# -------------------------
# Bar Chart
# -------------------------

st.subheader("Vehicle Distribution")

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

# -------------------------
# Pie Chart
# -------------------------

st.subheader("Vehicle Share")

pie_fig = px.pie(
    chart_df,
    names="Vehicle Type",
    values="Count"
)

st.plotly_chart(
    pie_fig,
    use_container_width=True
)

# -------------------------
# Historical Trend
# -------------------------

st.subheader("Historical Traffic Trend")

if len(history_df) > 0:

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

else:

    st.info(
        "No historical analytics data available."
    )

# -------------------------
# Insights
# -------------------------

st.subheader("AI Traffic Insights")

st.success(
    f"Most common vehicle detected: {most_common_vehicle}"
)

st.info(
    f"Traffic density currently: {traffic_level}"
)

if len(history_df) > 0:
    st.warning(
        f"Historical records stored: {len(history_df)}"
    )

# -------------------------
# CSV Download
# -------------------------

csv = df.to_csv(index=False)

st.download_button(
    label="📥 Download CSV Report",
    data=csv,
    file_name="vehicle_report.csv",
    mime="text/csv"
)

# -------------------------
# Traffic Alerts
# -------------------------

st.subheader("🚨 Traffic Alerts")

alert_file = "outputs/traffic_alerts.csv"

if os.path.exists(alert_file):

    alert_df = pd.read_csv(alert_file)

    st.dataframe(
        alert_df.tail(10),
        use_container_width=True
    )

else:

    st.info(
        "No traffic alerts available."
    )
```
