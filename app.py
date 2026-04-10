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
        st.selectbox(
            "Response Mode",
            options=["Detailed", "Concise", "Socratic"],
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
    st.caption(f"📄 Active document: **{st.session_state.current_file_name}**")

    # Process document if not already done
    if not st.session_state.document_processed:
        with st.spinner("Processing document..."):
            try:
                from pipeline.ingestion import process_document
                st.session_state.retriever = process_document(
                    st.session_state.uploaded_file
                )
                st.session_state.document_processed = True
                st.success("Document processed successfully! Ask me anything.")
            except Exception as e:
                st.error(f"Failed to process document: {str(e)}")
                return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from pipeline.qa import answer_question
                    response = answer_question(
                        question=prompt,
                        retriever=st.session_state.retriever,
                        chat_history=st.session_state.messages[:-1],
                        mode=st.session_state.get("response_mode", "Detailed"),
                    )
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)


def main():
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
