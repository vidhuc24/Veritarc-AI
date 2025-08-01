"""
AuditFlow AI - Streamlit Frontend
AI-Powered Control Evidence Validation System
"""

import streamlit as st
import requests
import json
from typing import Optional
import time

# Page configuration
st.set_page_config(
    page_title="AuditFlow AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .results-section {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 AuditFlow AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em; color: #666;">AI-Powered Control Evidence Validation System</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Framework selection
        framework = st.selectbox(
            "Compliance Framework",
            options=["SOX", "SOC2", "ISO27001"],
            help="Select the compliance framework for validation"
        )
        
        # Control selection based on framework
        control_options = {
            "SOX": ["SOX-ITGC-01: Logical Access Controls", "SOX-ITGC-02: Change Management", "SOX-ITGC-03: Data Backup & Recovery"],
            "SOC2": ["CC6.1: Logical and Physical Access Controls", "CC6.2: System Credentials", "CC6.3: Network Access"],
            "ISO27001": ["A.9.2.1: User Registration", "A.9.2.2: User Access Provisioning", "A.9.2.3: Management of Privileged Access"]
        }
        
        control = st.selectbox(
            "Control Requirement",
            options=control_options[framework],
            help="Select the specific control to validate against"
        )
        
        # Extract control ID
        control_id = control.split(":")[0]
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum confidence level for validation"
            )
            
            api_endpoint = st.text_input(
                "API Endpoint",
                value="http://localhost:8000",
                help="Backend API endpoint"
            )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Evidence")
        
        # File upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose evidence document",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
            help="Upload your control evidence document (PDF, DOCX, or image)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size:,} bytes")
            st.info(f"📋 Selected control: {control}")
            
            # Validation button
            if st.button("🚀 Validate Evidence", type="primary", use_container_width=True):
                validate_evidence(uploaded_file, control_id, framework, api_endpoint)
    
    with col2:
        st.header("📊 Validation Results")
        
        # Display results if available
        if 'validation_results' in st.session_state:
            display_results(st.session_state.validation_results)
        else:
            st.info("👆 Upload a document and click 'Validate Evidence' to see results")

def validate_evidence(uploaded_file, control_id: str, framework: str, api_endpoint: str):
    """Send file to backend for validation"""
    
    with st.spinner("🔍 Analyzing evidence document..."):
        try:
            # Prepare the request
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {
                "control_id": control_id,
                "framework": framework
            }
            
            # Make API request
            response = requests.post(
                f"{api_endpoint}/validate-evidence",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()
                st.session_state.validation_results = results
                st.success("✅ Validation completed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Validation failed: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Make sure the server is running on " + api_endpoint)
        except requests.exceptions.Timeout:
            st.error("⏰ Request timed out. The document might be too large or complex.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

def display_results(results: dict):
    """Display validation results in a nice format"""
    
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = results.get('validation_score', 0)
        st.metric(
            label="📈 Validation Score",
            value=f"{score:.1%}",
            delta=f"{score - 0.7:.1%}" if score > 0.7 else f"{score - 0.7:.1%}"
        )
    
    with col2:
        confidence = results.get('confidence_level', 'Unknown')
        st.metric(
            label="🎯 Confidence Level",
            value=confidence
        )
    
    with col3:
        status = results.get('compliance_status', 'Unknown')
        status_color = "🟢" if "Compliant" in status else "🟡" if "Partial" in status else "🔴"
        st.metric(
            label="✅ Compliance Status",
            value=f"{status_color} {status}"
        )
    
    # Assessment
    st.subheader("📝 Assessment")
    assessment = results.get('assessment', 'No assessment available')
    st.write(assessment)
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    
    # Gaps identified
    gaps = results.get('gaps_identified', [])
    if gaps:
        st.subheader("⚠️ Gaps Identified")
        for gap in gaps:
            st.warning(f"• {gap}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Export options
    st.subheader("📄 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Copy to Clipboard"):
            # Format results as text
            text_results = format_results_as_text(results)
            st.code(text_results, language="text")
    
    with col2:
        if st.button("💾 Download JSON"):
            st.download_button(
                label="Download Results",
                data=json.dumps(results, indent=2),
                file_name=f"validation_results_{int(time.time())}.json",
                mime="application/json"
            )

def format_results_as_text(results: dict) -> str:
    """Format results as readable text"""
    text = f"""
AuditFlow AI - Validation Results
================================

Validation Score: {results.get('validation_score', 0):.1%}
Confidence Level: {results.get('confidence_level', 'Unknown')}
Compliance Status: {results.get('compliance_status', 'Unknown')}

Assessment:
{results.get('assessment', 'No assessment available')}

Recommendations:
{chr(10).join(f"• {rec}" for rec in results.get('recommendations', []))}

Gaps Identified:
{chr(10).join(f"• {gap}" for gap in results.get('gaps_identified', []))}
"""
    return text.strip()

# About section
def show_about():
    st.header("ℹ️ About AuditFlow AI")
    
    st.markdown("""
    **AuditFlow AI** is an intelligent evidence validation system that helps audit teams:
    
    - 🚀 **Save Time**: Reduce manual evidence review by 70%
    - 🎯 **Improve Consistency**: Eliminate subjective evaluations
    - 📊 **Enhance Quality**: Identify gaps and provide recommendations
    - 🔍 **Ensure Compliance**: Support SOX, SOC2, and ISO27001 frameworks
    
    ### How it works:
    1. Upload your evidence document (PDF, DOCX, or image)
    2. Select the compliance framework and control
    3. Get instant AI-powered validation with recommendations
    
    ### Supported Frameworks:
    - **SOX**: Sarbanes-Oxley Act compliance
    - **SOC2**: Service Organization Control 2
    - **ISO27001**: Information Security Management
    """)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "ℹ️ About"],
    index=0
)

if page == "🏠 Home":
    main()
elif page == "ℹ️ About":
    show_about()

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">Built with ❤️ using Streamlit and FastAPI | AuditFlow AI v1.0</p>',
    unsafe_allow_html=True
) 