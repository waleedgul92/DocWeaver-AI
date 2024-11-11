import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import tempfile
import os
import io
from PIL import Image, ImageDraw, ImageFont

# Configure the Streamlit page
st.set_page_config("MedQuery AI", initial_sidebar_state="collapsed")

# Custom CSS for animations and styling
st.markdown(
    """
    <style>
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateX(-20px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    .welcome-message {
        font-size: 48px; /* Adjust font size */
        text-align: center;
        color: white; /* Text color */
        animation: fadeIn 2.0s ease forwards; /* Apply animation */
        margin-top: 50px; /* Space above the message */
    }

    .stTextInput > div > input {
        height: 100%;  /* Adjust height */
        width: 100%;   /* Adjust width */
        font-size: 14px;  /* Adjust font size */
    }

    div[data-testid="column"] {
        width: fit-content !important;
        flex: unset;
        padding-left: 20px;  /* Left padding */
    }

    div[data-testid="column"] * {
        width: fit-content !important;
        vertical-align: left;  /* Align items to the left */
    }

    /* Styling the file uploader to match button styles */
    .stFileUploader {
        width: 100%;  /* Make the file uploader full width */
        border-radius: 5px;  /* Rounded corners */
        text-align: center;  /* Center text */
    }

    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
    }

    [data-testid='stFileUploader'] section > input + div {
        display: none;  /* Hide the default uploader text */
    }

    /* Animation for sidebar items */
    .sidebar .stSelectbox, 
    .sidebar .stButton, 
    .sidebar .stFileUploader {
        animation: fadeIn 1s ease forwards; /* Apply animation to sidebar elements */
    }

    .sidebar {
        animation: fadeIn 1s ease forwards; /* Apply animation to sidebar itself */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the animated welcome message
st.markdown("<div class='welcome-message'>Welcome to MedQuery AI!</div>", unsafe_allow_html=True)

# Custom CSS for spinner
st.markdown("""
  <style>
  div.stSpinner {
    text-align:center;
    align-items: center;
    justify-content: center;
  }
  </style>""", unsafe_allow_html=True)

# Placeholder for main content
main_placeholder = st.empty()

# Input text box for user query with selected model display
col_input, col_model = st.columns([3, 1])

with col_input:
    user_query = main_placeholder.text_input(
        "Question: ",
        placeholder="Type your question here for analyzing PDF or URL",
        max_chars=1000, label_visibility="hidden"
    )

# Create columns for action buttons
col1, col2, col3, col4 = st.columns([2.5, 2, 2.4, 3.7], vertical_alignment="center", gap="small")

with col2:
    pdf_analyze_button = st.button('Analyze PDF', key='analyze_text', help='Click to analyze the pdf.')

with col3:
    url_analyze_button=st.button('Analyze URL', key='Analyze URL', help='Click to analyze the url.')

# Sidebar elements with titles and selectboxes
st.sidebar.title("Model for Document Analysis")
selected_pdf_model = st.sidebar.selectbox("Select a Document Analysis", ["Gemini-1.5", "Gpt-2"], label_visibility="hidden")

st.sidebar.title("Model for URL Analysis")
selected_url_model = st.sidebar.selectbox("Select a URL Analysis", ["Model 1", "Model 2"], label_visibility="hidden")

st.sidebar.title("Upload a PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload an PDF File", type=["pdf"],
                                                  help="Upload an image for analysis ", label_visibility="hidden")

st.sidebar.title("Add a URL")
uploaded_url=st.sidebar.text_input("Enter the URL", help="Enter the URL for analysis", label_visibility="hidden")

with col_model:
    st.markdown(f"<div style='color:gray; font-size:14px text-align: right;'  > PDF Model: {selected_pdf_model}</div>", unsafe_allow_html=True)
st.markdown("""
  <style>
    .css-13sdm1b.e16nr0p33 {
      margin-top: -75px;
    }
  </style>
""", unsafe_allow_html=True)


# Generate and display image if button is clicked
if user_query and generate_image_button:
    with col_model:
        st.markdown(f"<div style='color:gray; font-size:14px text-align: right;'  > URL Model: {selected_url_model}</div>", unsafe_allow_html=True)
        st.markdown("""
        <style>
            .css-13sdm1b.e16nr0p33 {
            margin-top: -75px;
            }
        </style>
    """, unsafe_allow_html=True)
# Analyze text button logic (if needed)