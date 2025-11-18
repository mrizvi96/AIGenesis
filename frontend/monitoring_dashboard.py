"""
Real-time Monitoring Dashboard for AI Insurance Claims Processing System
"""
import streamlit as st
import requests
import time
import json
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="System Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.status-green { color: #10b981; }
.status-yellow { color: #f59e0b; }
.status-red { color: #ef4444; }
.system-health {
    background: #1f2937;
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# API base URL
API_BASE = "http://localhost:8000"

# Auto-refresh functionality
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

def get_system_health():
    """Get system health status"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_qdrant_status():
    """Get Qdrant database status"""
    try:
        response = requests.get(f"{API_BASE}/collections", timeout=5)
        if response.status_code == 200:
            collections = response.json().get("collections", [])
            return {"status": "healthy", "collections": collections, "count": len(collections)}
        else:
            return {"status": "error", "message": "API Error"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def simulate_metrics():
    """Simulate system metrics for demo"""
    import random

    # Generate realistic metrics
    current_time = datetime.now()

    # Processing times (ms)
    processing_times = [random.randint(200, 800) for _ in range(20)]

    # Claims processed
    claims_today = random.randint(45, 89)
    claims_this_hour = random.randint(3, 12)

    # System resources
    cpu_usage = random.uniform(15, 45)
    memory_usage = random.uniform(30, 60)
    disk_usage = random.uniform(25, 55)

    # API performance
    api_response_time = random.uniform(120, 350)
    api_success_rate = random.uniform(95, 99.5)

    # Database performance
    db_query_time = random.uniform(45, 180)
    db_connection_pool = random.uniform(60, 85)

    # Fraud detection
    fraud_detected_today = random.randint(2, 7)
    false_positive_rate = random.uniform(2.5, 8.5)

    # User activity
    active_users = random.randint(1, 5)
    concurrent_requests = random.randint(0, 3)

    return {
        "processing_times": processing_times,
        "claims_today": claims_today,
        "claims_this_hour": claims_this_hour,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "disk_usage": disk_usage,
        "api_response_time": api_response_time,
        "api_success_rate": api_success_rate,
        "db_query_time": db_query_time,
        "db_connection_pool": db_connection_pool,
        "fraud_detected_today": fraud_detected_today,
        "false_positive_rate": false_positive_rate,
        "active_users": active_users,
        "concurrent_requests": concurrent_requests,
        "uptime": "99.9%",
        "last_updated": current_time.strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    """Main dashboard application"""

    # Header
    st.title("🏥 AI Claims Processing System")
    st.markdown("### Real-Time Monitoring Dashboard")
    st.markdown("---")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

    # System Status Overview
    col1, col2, col3 = st.columns(3)

    with col1:
        system_health = get_system_health()
        status_color = "status-green" if system_health.get("status") == "healthy" else "status-red"
        st.markdown(f"""
        <div class="system-health">
            <h4>System Status</h4>
            <h2 class="{status_color}">● {system_health.get("status", "Unknown").upper()}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        qdrant_status = get_qdrant_status()
        st.markdown(f"""
        <div class="system-health">
            <h4>Database Status</h4>
            <h2 class="status-green">● {qdrant_status.get("status", "Unknown").upper()}</h2>
            <p>{qdrant_status.get("count", 0)} Collections</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        uptime = "99.9%"
        st.markdown(f"""
        <div class="system-health">
            <h4>System Uptime</h4>
            <h2 class="status-green">● {uptime}</h2>
            <p>Last 30 days</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Get metrics
    metrics = simulate_metrics()

    # Key Performance Indicators
    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics["claims_today"]}</h3>
            <p>Claims Processed Today</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics["api_response_time"]:.0f}ms</h3>
            <p>Avg API Response</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics["fraud_detected_today"]}</h3>
            <p>Fraud Cases Detected</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics["api_success_rate"]:.1f}%</h3>
            <p>API Success Rate</p>
        </div>
        """, unsafe_allow_html=True)

    # Charts Section
    st.markdown("---")
    st.subheader("📈 Performance Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Processing Time Distribution
        fig = go.Figure(data=[go.Histogram(x=metrics["processing_times"], nbinsx=10)])
        fig.update_layout(
            title="Processing Time Distribution",
            xaxis_title="Processing Time (ms)",
            yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # System Resources
        resources = {
            "Resource": ["CPU", "Memory", "Disk"],
            "Usage": [metrics["cpu_usage"], metrics["memory_usage"], metrics["disk_usage"]]
        }
        df_resources = pd.DataFrame(resources)

        fig = px.bar(
            df_resources,
            x="Resource",
            y="Usage",
            title="System Resource Usage",
            height=300
        )
        fig.update_layout(yaxis_title="Usage (%)")
        st.plotly_chart(fig, use_container_width=True)

    # Detailed Metrics
    st.markdown("---")
    st.subheader("🔧 System Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**API Performance**")
        st.write(f"• Response Time: {metrics['api_response_time']:.1f}ms")
        st.write(f"• Success Rate: {metrics['api_success_rate']:.1f}%")
        st.write(f"• Concurrent Requests: {metrics['concurrent_requests']}")

        st.write("**Database Performance**")
        st.write(f"• Query Time: {metrics['db_query_time']:.1f}ms")
        st.write(f"• Connection Pool: {metrics['db_connection_pool']:.1f}%")

    with col2:
        st.write("**Claims Processing**")
        st.write(f"• Today: {metrics['claims_today']} claims")
        st.write(f"• This Hour: {metrics['claims_this_hour']} claims")
        st.write(f"• Avg Processing: {sum(metrics['processing_times'])/len(metrics['processing_times']):.0f}ms")

        st.write("**Fraud Detection**")
        st.write(f"• Detected Today: {metrics['fraud_detected_today']}")
        st.write(f"• False Positive Rate: {metrics['false_positive_rate']:.1f}%")

    with col3:
        st.write("**User Activity**")
        st.write(f"• Active Users: {metrics['active_users']}")
        st.write(f"• Current Session: 1")

        st.write("**System Health**")
        st.write(f"• Last Updated: {metrics['last_updated']}")
        st.write(f"• Auto Refresh: {refresh_interval}s")

        # Quick action buttons
        if st.button("Force Refresh"):
            st.rerun()

        if st.button("View Logs"):
            st.info("Logs feature coming soon!")

    # Status Messages
    with st.expander("📢 System Status Messages"):
        st.success("✅ All systems operational")
        st.info("ℹ️ High claim volume detected - performing optimally")
        st.warning("⚠️ Scheduled maintenance at 2:00 AM UTC")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>AI Insurance Claims Processing System | Real-time Monitoring Dashboard</p>
        <p>Powered by Qdrant Vector Database & Advanced AI Technologies</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()