"""
Streamlit Frontend for AI Insurance Claims Processing
Multimodal interface for claim submission and analysis
"""

import os
import json
import time
import requests
import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Page configuration
st.set_page_config(
    page_title="AI Insurance Claims Processor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def api_call(endpoint: str, method: str = "GET", data: Dict[str, Any] = None, files: Dict = None) -> Dict[str, Any]:
    """Make API call to backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"

        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, timeout=60)
            else:
                response = requests.post(url, json=data, timeout=30)

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {"success": False, "error": str(e)}

def check_api_health():
    """Check if API is healthy"""
    try:
        response = api_call("/health")
        # Health endpoint returns {"status": "healthy", ...} not {"success": true}
        return response.get("status") == "healthy"
    except:
        return False

def display_success_message(message: str):
    """Display success message with custom styling"""
    st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)

def display_warning_message(message: str):
    """Display warning message with custom styling"""
    st.markdown(f'<div class="warning-box">⚠️ {message}</div>', unsafe_allow_html=True)

def display_error_message(message: str):
    """Display error message with custom styling"""
    st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)

def display_metric(label: str, value: str, delta: Optional[str] = None):
    """Display metric with custom styling"""
    if delta:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #6c757d;">{label}</h3>
            <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{value}</p>
            <p style="margin: 0; color: #28a745; font-size: 0.9rem;">{delta}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #6c757d;">{label}</h3>
            <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{value}</p>
        </div>
        """, unsafe_allow_html=True)

def claim_submission_page():
    """Claim submission page"""
    st.markdown('<h1 class="main-header">📝 Submit New Claim</h1>', unsafe_allow_html=True)

    # Two column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Claim Information")

        # Claim form
        with st.form("claim_form"):
            claim_id = st.text_input("Claim ID (Optional)", placeholder="Auto-generated if empty")
            customer_id = st.text_input("Customer ID *", placeholder="e.g., CUST_123456")
            policy_number = st.text_input("Policy Number *", placeholder="e.g., POL_789012")

            claim_type = st.selectbox("Claim Type *", [
                "auto", "home", "health", "life", "property", "liability", "other"
            ])

            description = st.text_area(
                "Claim Description *",
                placeholder="Please provide a detailed description of the incident...",
                height=150
            )

            amount = st.number_input(
                "Claim Amount ($) *",
                min_value=0.0,
                step=100.0,
                format="%.2f"
            )

            location = st.text_input("Location", placeholder="e.g., New York, NY")

            submitted = st.form_submit_button("Submit Claim", type="primary")

            if submitted:
                # Validate required fields
                if not customer_id or not policy_number or not description or amount <= 0:
                    st.error("Please fill in all required fields marked with *")
                    return

                # Prepare claim data
                claim_data = {
                    "claim_id": claim_id if claim_id else None,
                    "customer_id": customer_id,
                    "policy_number": policy_number,
                    "claim_type": claim_type,
                    "description": description,
                    "amount": float(amount),
                    "location": location
                }

                # Submit claim with proper structure
                submission_data = {
                    "claim_data": claim_data,
                    "text_data": description
                }

                # Submit claim
                with st.spinner("Processing claim..."):
                    result = api_call("/submit_claim", "POST", data=submission_data)

                if result.get("success"):
                    display_success_message("Claim submitted successfully!")
                    st.session_state.last_claim_result = result
                else:
                    display_error_message(f"Failed to submit claim: {result.get('error', 'Unknown error')}")

    with col2:
        st.subheader("Upload Supporting Documents")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload supporting documents (images, audio, video)",
            type=['jpg', 'jpeg', 'png', 'gif', 'mp3', 'wav', 'mp4', 'avi'],
            help="Supported formats: Images (JPG, PNG), Audio (MP3, WAV), Video (MP4, AVI)"
        )

        if uploaded_file:
            st.info(f"File selected: {uploaded_file.name}")
            st.info(f"File size: {uploaded_file.size / 1024:.2f} KB")

            # Detect file type
            file_type = None
            if uploaded_file.type.startswith('image/'):
                file_type = "image"
            elif uploaded_file.type.startswith('audio/'):
                file_type = "audio"
            elif uploaded_file.type.startswith('video/'):
                file_type = "video"

            if file_type:
                if st.button("Process File", type="secondary"):
                    # Prepare data for file processing
                    file_data = {
                        'file': uploaded_file,
                        'claim_id': claim_id if claim_id else f"FILE_{int(time.time())}",
                        'customer_id': customer_id,
                        'claim_type': claim_type,
                        'description': description,
                        'amount': str(amount),
                        'modality': file_type
                    }

                    with st.spinner("Processing file..."):
                        result = api_call("/process_file", "POST", files=file_data)

                    if result.get("success"):
                        display_success_message(f"{file_type.capitalize()} file processed successfully!")
                        st.session_state.last_file_result = result
                    else:
                        display_error_message(f"Failed to process file: {result.get('error', 'Unknown error')}")

        # Recent results
        if 'last_claim_result' in st.session_state or 'last_file_result' in st.session_state:
            st.subheader("Recent Submissions")

            if 'last_claim_result' in st.session_state:
                result = st.session_state.last_claim_result
                if result.get("success"):
                    claim_id = result["data"]["claim_id"]
                    st.markdown(f"**Claim ID:** {claim_id}")
                    st.markdown(f"**Status:** Submitted")

            if 'last_file_result' in st.session_state:
                result = st.session_state.last_file_result
                if result.get("success"):
                    file_info = result["data"]["file_info"]
                    st.markdown(f"**File:** {file_info['filename']}")
                    st.markdown(f"**Type:** {file_info['modality']}")

def recommendation_page():
    """Recommendation page"""
    st.markdown('<h1 class="main-header">🤖 Get AI Recommendation</h1>', unsafe_allow_html=True)

    # Quick analysis form
    with st.form("recommendation_form"):
        st.subheader("Claim Analysis Request")

        customer_id = st.text_input("Customer ID *", placeholder="e.g., CUST_123456")
        policy_number = st.text_input("Policy Number *", placeholder="e.g., POL_789012")

        claim_type = st.selectbox("Claim Type *", [
            "auto", "home", "health", "life", "property", "liability", "other"
        ])

        description = st.text_area(
            "Claim Description *",
            placeholder="Please provide a detailed description of the incident...",
            height=150
        )

        amount = st.number_input(
            "Claim Amount ($) *",
            min_value=0.0,
            step=100.0,
            format="%.2f"
        )

        location = st.text_input("Location", placeholder="e.g., New York, NY")

        analyze_button = st.form_submit_button("🔍 Get Recommendation", type="primary")

        if analyze_button:
            # Validate required fields
            if not customer_id or not policy_number or not description or amount <= 0:
                st.error("Please fill in all required fields marked with *")
                return

            # Prepare recommendation request
            claim_data = {
                "claim_id": f"REC_{int(time.time())}",
                "customer_id": customer_id,
                "policy_number": policy_number,
                "claim_type": claim_type,
                "description": description,
                "amount": float(amount),
                "location": location
            }

            # Get recommendation
            with st.spinner("Generating AI recommendation..."):
                result = api_call("/get_recommendation", "POST", data={"claim_data": claim_data})

            if result.get("success"):
                display_success_message("Recommendation generated successfully!")
                st.session_state.last_recommendation = result["data"]
            else:
                display_error_message(f"Failed to generate recommendation: {result.get('error', 'Unknown error')}")

    # Display recommendation results
    if 'last_recommendation' in st.session_state:
        rec = st.session_state.last_recommendation

        st.markdown("---")
        st.subheader("🎯 AI Recommendation Results")

        # Main recommendation
        col1, col2 = st.columns(2)

        with col1:
            recommendation = rec.get("recommendation", {})
            display_metric(
                "Recommended Action",
                recommendation.get("action", "N/A"),
                recommendation.get("reason", "")
            )
            display_metric(
                "Priority",
                recommendation.get("priority", "N/A"),
                f"Processing time: {recommendation.get('estimated_processing_time', 'N/A')}"
            )

        with col2:
            fraud_risk = rec.get("fraud_risk", {})
            risk_level = fraud_risk.get("risk_level", "N/A")
            risk_score = fraud_risk.get("risk_score", 0)

            risk_color = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
            display_metric(
                f"Fraud Risk {risk_color}",
                f"{risk_level} ({risk_score:.1%})",
                f"Similar claims: {rec.get('similar_claims_count', 0)}"
            )

        # Settlement estimate
        settlement = rec.get("settlement_estimate", {})
        if settlement:
            st.subheader("💰 Settlement Estimate")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Estimated Amount", f"${settlement.get('estimated_amount', 0):,.2f}")
            with col2:
                confidence = settlement.get('confidence_percentage', 0)
                st.metric("Confidence", f"{confidence:.1f}%")
            with col3:
                range_min, range_max = settlement.get('confidence_range', [0, 0])
                st.metric("Range", f"${range_min:,.2f} - ${range_max:,.2f}")

            if settlement.get('factors'):
                st.info("Factors affecting estimate: " + ", ".join(settlement['factors']))

        # Confidence scores
        confidence = rec.get("confidence_scores", {})
        if confidence:
            st.subheader("📊 Confidence Scores")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Data Confidence", f"{confidence.get('data_confidence', 0):.1%}")
                st.metric("Model Confidence", f"{confidence.get('model_confidence', 0):.1%}")

            with col2:
                st.metric("Pattern Confidence", f"{confidence.get('pattern_confidence', 0):.1%}")
                st.metric("Overall Confidence", f"{confidence.get('overall_confidence', 0):.1%}")

        # Similar claims summary
        similar_claims = rec.get("similar_claims_summary", {})
        if similar_claims.get("top_claims"):
            st.subheader("🔍 Similar Claims Found")

            for claim in similar_claims["top_claims"]:
                with st.expander(f"Claim {claim.get('claim_id', 'Unknown')} (Similarity: {claim.get('similarity', 0):.1%})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Status:** {claim.get('status', 'N/A')}")
                        st.write(f"**Amount:** ${claim.get('amount', 0):,.2f}")

                    with col2:
                        st.write(f"**Similarity:** {claim.get('similarity', 0):.1%}")
                        st.write(f"**Description:** {claim.get('description', 'N/A')}")

def search_page():
    """Search page"""
    st.markdown('<h1 class="main-header">🔍 Search Similar Claims</h1>', unsafe_allow_html=True)

    # Search form
    with st.form("search_form"):
        query = st.text_input(
            "Search Query *",
            placeholder="Describe the claim or incident...",
            help="Enter keywords or a description to find similar claims"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            modality = st.selectbox("Search Modality", [
                "text_claims", "image_claims", "audio_claims", "video_claims"
            ])

        with col2:
            limit = st.number_input("Max Results", min_value=1, max_value=20, value=5)

        with col3:
            cross_modal = st.checkbox("Cross-Modal Search", help="Search across multiple modalities")

        search_button = st.form_submit_button("🔎 Search", type="primary")

        if search_button:
            if not query:
                st.error("Please enter a search query")
                return

            if cross_modal:
                # Cross-modal search
                modalities = "text_claims,image_claims,audio_claims,video_claims"
                with st.spinner("Performing cross-modal search..."):
                    result = api_call("/search_claims", "POST", data={
                        "query": query,
                        "modalities": modalities
                    })

                if result.get("success"):
                    display_success_message(f"Found {result['data']['total_results']} results across modalities")
                    st.session_state.search_results = result["data"]["results"]
                    st.session_state.search_query = query
                else:
                    display_error_message(f"Search failed: {result.get('error', 'Unknown error')}")
            else:
                # Single modality search
                search_data = {
                    "query": query,
                    "modality": modality,
                    "limit": limit
                }

                with st.spinner("Searching similar claims..."):
                    result = api_call("/search_claims", "POST", data=search_data)

                if result.get("success"):
                    similar_claims = result["data"]["similar_claims"]
                    display_success_message(f"Found {len(similar_claims)} similar claims")
                    st.session_state.search_results = similar_claims
                    st.session_state.search_query = query
                else:
                    display_error_message(f"Search failed: {result.get('error', 'Unknown error')}")

    # Display search results
    if 'search_results' in st.session_state:
        st.markdown("---")
        st.subheader(f"📋 Search Results for: '{st.session_state.get('search_query', '')}'")

        results = st.session_state.search_results

        if isinstance(results, dict):  # Cross-modal results
            for modality, claims in results.items():
                if claims:
                    st.write(f"### {modality.replace('_', ' ').title()} ({len(claims)} results)")

                    for i, claim in enumerate(claims):
                        with st.expander(f"Result {i+1}: {claim.get('claim_id', 'Unknown')} (Score: {claim.get('similarity_score', 0):.1%})"):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write(f"**Status:** {claim.get('status', 'N/A')}")
                                st.write(f"**Amount:** ${claim.get('amount', 0):,.2f}")
                                st.write(f"**Similarity:** {claim.get('similarity_score', 0):.1%}")

                            with col2:
                                st.write(f"**Customer:** {claim.get('customer_id', 'N/A')}")
                                st.write(f"**Date:** {claim.get('date_submitted', 'N/A')}")
                                st.write(f"**Type:** {claim.get('claim_type', 'N/A')}")

                            if claim.get('description'):
                                st.write(f"**Description:** {claim['description']}")
        else:  # Single modality results
            if results:
                for i, claim in enumerate(results):
                    with st.expander(f"Result {i+1}: {claim.get('claim_id', 'Unknown')} (Score: {claim.get('similarity_score', 0):.1%})"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Status:** {claim.get('status', 'N/A')}")
                            st.write(f"**Amount:** ${claim.get('amount', 0):,.2f}")
                            st.write(f"**Similarity:** {claim.get('similarity_score', 0):.1%}")

                        with col2:
                            st.write(f"**Customer:** {claim.get('customer_id', 'N/A')}")
                            st.write(f"**Date:** {claim.get('date_submitted', 'N/A')}")
                            st.write(f"**Type:** {claim.get('claim_type', 'N/A')}")

                        if claim.get('description'):
                            st.write(f"**Description:** {claim['description']}")
            else:
                st.info("No results found")

def dashboard_page():
    """Dashboard page"""
    st.markdown('<h1 class="main-header">📊 System Dashboard</h1>', unsafe_allow_html=True)

    # Check API health first
    if not check_api_health():
        display_error_message("Cannot connect to the API server. Please make sure the backend is running on localhost:8000")
        return

    # System overview
    st.subheader("🖥️ System Overview")

    try:
        # Get system health status
        health_info = api_call("/health")
        if health_info.get("status") == "healthy":
            data = health_info

            col1, col2, col3 = st.columns(3)

            with col1:
                status = data["status"]
                services = data["services"]
                display_metric("API Status", status.upper(), f"Qdrant: {'Connected' if services['qdrant_connected'] else 'Disconnected'}")

            with col2:
                embedder_ready = services["embedder_loaded"]
                model_status = "Loaded" if embedder_ready else "Not Loaded"
                display_metric("Embedding Model", model_status, f"Recommender: {'Ready' if services['recommender_ready'] else 'Not Ready'}")

            with col3:
                version = data["version"]
                timestamp = data["timestamp"].split("T")[1][:8]
                date = data["timestamp"].split("T")[0]
                display_metric("Version", version, f"Updated: {timestamp}")

        # Collections overview
        st.subheader("📁 Vector Collections")

        collections_info = api_call("/collections")
        if collections_info.get("success"):
            collections = collections_info["data"]

            # Create metrics for collections
            cols = st.columns(3)

            collection_list = list(collections.keys())
            for i, col in enumerate(cols):
                with col:
                    if i < len(collection_list):
                        collection_name = collection_list[i]
                        collection_data = collections[collection_name]

                        if "error" not in collection_data:
                            vectors = collection_data.get("vectors_count", 0)
                            status = collection_data.get("status", "unknown")
                            display_metric(collection_name.replace("_", " ").title(), f"{vectors or 'Empty'}", f"Status: {status}")

    except Exception as e:
        display_error_message(f"Failed to load system information: {str(e)}")

    # API documentation
    st.markdown("---")
    st.subheader("📚 API Documentation")

    api_docs = """
    ### Available API Endpoints

    1. **Health Check**
       - `GET /health` - Check API health status

    2. **Claim Operations**
       - `POST /submit_claim` - Submit a new claim
       - `POST /get_recommendation` - Get AI recommendation
       - `POST /process_file` - Process uploaded files

    3. **Search Operations**
       - `POST /search_claims` - Search similar claims
       - `POST /search_claims` - Cross-modal search

    4. **System Information**
       - `GET /collections` - Get collection info
       - `GET /` - Get system status

    ### Getting Started
    1. Make sure the backend server is running on `localhost:8000`
    2. Use the **Submit New Claim** page to add claims
    3. Use the **Get AI Recommendation** page to analyze claims
    4. Use the **Search Similar Claims** page to find related claims
    """

    st.markdown(api_docs)

    # Add button to open advanced monitoring dashboard
    st.markdown("---")
    if st.button("🚀 Open Advanced Monitoring Dashboard", type="primary", use_container_width=True):
        st.info("Opening the advanced monitoring dashboard in a new tab...")
        st.markdown("""
        <script>
        window.open('http://localhost:8502', '_blank');
        </script>
        """, unsafe_allow_html=True)

    st.markdown("**To start the monitoring dashboard:**")
    st.code("streamlit run monitoring_dashboard.py --server.port 8502")

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("🏥 AI Claims Processor")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Select Page",
        ["📝 Submit New Claim", "🤖 Get AI Recommendation", "🔍 Search Similar Claims", "📊 Dashboard"],
        key="page_selector"
    )

    # API status indicator in sidebar
    api_status = check_api_health()
    if api_status:
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Disconnected")
        st.sidebar.info("Please start the backend server:")
        st.sidebar.code("cd backend && python main.py")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **AI Insurance Claims Processor**

    Multimodal AI-powered insurance claims processing using:
    - Qdrant vector search
    - Sentence transformers
    - Google Vision API
    - OpenAI Whisper API
    """)

    # Display selected page
    if page == "📝 Submit New Claim":
        claim_submission_page()
    elif page == "🤖 Get AI Recommendation":
        recommendation_page()
    elif page == "🔍 Search Similar Claims":
        search_page()
    elif page == "📊 Dashboard":
        dashboard_page()

if __name__ == "__main__":
    main()