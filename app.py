"""DeepTutor - AI-powered document tutoring application.

Main entry point for the Streamlit-based web application.
Fork of HKUDS/DeepTutor with enhancements.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DeepTutor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = None


def render_sidebar():
    """Render the sidebar with file upload and settings."""
    with st.sidebar:
        st.title("📚 DeepTutor")
        st.markdown("*Your AI-powered document tutor*")
        st.divider()

        # File upload section
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to start learning",
        )

        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.current_file_name:
                st.session_state.uploaded_file = uploaded_file
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.document_processed = False
                st.session_state.messages = []
                st.session_state.retriever = None

        st.divider()

        # Settings section
        st.subheader("Settings")
        # Defaulting to Socratic mode — I find it much better for actually learning
        # material rather than just getting answers handed to you.
        st.selectbox(
            "Response Mode",
            options=["Detailed", "Concise", "Socratic"],
            index=2,
            key="response_mode",
            help="Choose how DeepTutor responds to your questions",
        )

        st.divider()
        st.caption("DeepTutor v1.0.0")
        st.caption("Fork of [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor)")


def render_main_content():
    """Render the main chat interface."""
    st.title("DeepTutor Chat")

    if st.session_state.uploaded_file is None:
        # Welcome screen
        st.info(
            "👈 **Get started by uploading a PDF document** in the sidebar.\n\n"
            "DeepTutor will help you understand and learn from your documents "
            "through interactive Q&A."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Documents Supported", "PDF")
        with col2:
            st.metric("🧠 AI Models", "Multiple")
        with col3:
            st.metric("💬 Interaction Mode", "Chat")
        return

    # Display current document info
    st.c