"""
AuditFlow AI - Streamlit Frontend
AI-Powered Control Evidence Validation System with Document Chat
"""

import streamlit as st
import requests
import json
from typing import Optional, List
import time
import uuid

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
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .chat-input {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_document' not in st.session_state:
    st.session_state.uploaded_document = None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 AuditFlow AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em; color: #666;">AI-Powered Control Evidence Validation & Document Chat System</p>', unsafe_allow_html=True)
    
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
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["📊 Evidence Validation", "💬 Document Chat"])
    
    with tab1:
        validation_interface(control, control_id, framework, api_endpoint)
    
    with tab2:
        chat_interface(api_endpoint)

def validation_interface(control: str, control_id: str, framework: str, api_endpoint: str):
    """Evidence validation interface"""
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Evidence")
        
        # File upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose evidence document",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
            help="Upload your control evidence document (PDF, DOCX, or image)",
            key="validation_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Store uploaded document for chat
            st.session_state.uploaded_document = uploaded_file
            
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

def chat_interface(api_endpoint: str):
    """Document chat interface"""
    
    st.header("💬 Chat with Your Document")
    
    # Check if document is uploaded
    if st.session_state.uploaded_document is None:
        st.info("📄 Please upload a document in the 'Evidence Validation' tab first to start chatting.")
        return
    
    # Document info
    st.success(f"📄 Chatting with: {st.session_state.uploaded_document.name}")
    
    # Chat history container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>AI Assistant:</strong> {message['content']}
                    <br><small><em>Sources: {', '.join(message.get('sources', []))}</em></small>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    # Suggested questions
    st.subheader("💡 Suggested Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What does this document show about access controls?"):
            send_chat_message("What does this document show about access controls?", api_endpoint)
    
    with col2:
        if st.button("Are there any approval processes documented?"):
            send_chat_message("Are there any approval processes documented?", api_endpoint)
    
    with col3:
        if st.button("What compliance requirements does this address?"):
            send_chat_message("What compliance requirements does this address?", api_endpoint)
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question about your document:",
            placeholder="e.g., What users are mentioned in this access review?",
            help="Ask any question about the uploaded document"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit_button = st.form_submit_button("Send 💬", type="primary")
        with col2:
            clear_button = st.form_submit_button("Clear Chat 🗑️")
        
        if submit_button and user_input:
            send_chat_message(user_input, api_endpoint)
        
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()

def send_chat_message(message: str, api_endpoint: str):
    """Send chat message to backend"""
    
    with st.spinner("🤔 AI is thinking..."):
        try:
            # Prepare chat request
            chat_data = {
                "message": message,
                "document_id": st.session_state.uploaded_document.name if st.session_state.uploaded_document else None,
                "conversation_id": st.session_state.conversation_id
            }
            
            # Make API request
            response = requests.post(
                f"{api_endpoint}/chat",
                json=chat_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add messages to chat history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': message
                })
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['response'],
                    'sources': result.get('sources', []),
                    'confidence': result.get('confidence', 0)
                })
                
                # Update conversation ID
                st.session_state.conversation_id = result['conversation_id']
                
                st.rerun()
            else:
                st.error(f"❌ Chat failed: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Make sure the server is running on " + api_endpoint)
        except requests.exceptions.Timeout:
            st.error("⏰ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

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
    **AuditFlow AI** is an intelligent evidence validation system with document chat capabilities that helps audit teams:
    
    - 🚀 **Save Time**: Reduce manual evidence review by 70%
    - 🎯 **Improve Consistency**: Eliminate subjective evaluations
    - 📊 **Enhance Quality**: Identify gaps and provide recommendations
    - 💬 **Interactive Analysis**: Chat with documents to extract insights
    - 🔍 **Ensure Compliance**: Support SOX, SOC2, and ISO27001 frameworks
    
    ### How it works:
    
    **Evidence Validation:**
    1. Upload your evidence document (PDF, DOCX, or image)
    2. Select the compliance framework and control
    3. Get instant AI-powered validation with recommendations
    
    **Document Chat:**
    1. Upload a document in the validation tab
    2. Switch to the chat tab
    3. Ask natural language questions about your document
    4. Get AI-powered answers with source references
    
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
    '<p style="text-align: center; color: #666;">Built with ❤️ using Streamlit and FastAPI | AuditFlow AI v1.0 with Document Chat</p>',
    unsafe_allow_html=True
) 