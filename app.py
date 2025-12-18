from dotenv import load_dotenv
import streamlit as st
import os
from pypdf import PdfReader
import google.generativeai as genai

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("⚠️ Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()
else:
    genai.configure(api_key=API_KEY)

# ---------------------- Utility Functions ----------------------

def get_gemini_response(system_prompt, resume_text, job_desc):
    """Send resume text + job description to Gemini with fallback."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        model = genai.GenerativeModel("gemini-2.0-pro")

    prompt = f"""
    {system_prompt}

    =====================
    RESUME:
    =====================
    {resume_text}

    =====================
    JOB DESCRIPTION:
    =====================
    {job_desc}
    """

    response = model.generate_content(prompt)
    return response.text


def input_pdf_setup(uploaded_file):
    """Extract text from PDF using pypdf (NO Poppler required)."""
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")

    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="AI Powered Resume Analyzer", page_icon="📄", layout="centered")

# Custom CSS (UNCHANGED)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    body {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #f0f4, #406078);
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-weight: 500;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #132743;
        color: yellow;
    }
</style>
""", unsafe_allow_html=True)

st.header("AI-Powered Resume Analyzer")
input_text = st.text_area("Job Description and Demands:", key="input")
uploaded_file = st.file_uploader("Upload your resume here (PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF Uploaded Successfully")

col1, col2, col3 = st.columns(3)

with col1:
    submit1 = st.button("Request Improvements")
with col2:
    submit2 = st.button("Percentage match")
with col3:
    submit3 = st.button("Evaluate Resume")

# ---------------------- PROMPTS ----------------------

input_prompt1 = """
You are an experienced Technical Human Resource Manager.
Review the resume against the job description.
Highlight flaws, weaknesses, and improvement areas.
"""

input_prompt2 = """
You are a strict ATS scanner.
First output the percentage match.
Then list missing keywords.
Finally give a verdict.
"""

input_prompt3 = """
You are an experienced HR Manager.
Evaluate alignment with the role.
Highlight strengths and weaknesses.
"""

# ---------------------- SUBMIT ACTIONS ----------------------

if submit1:
    if uploaded_file:
        resume_text = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, resume_text, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.error("Please upload the resume")

elif submit2:
    if uploaded_file:
        resume_text = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt2, resume_text, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.error("Please upload the resume")

elif submit3:
    if uploaded_file:
        resume_text = input_pdf_setup(uploaded_file)
        response1 = get_gemini_response(input_prompt3, resume_text, input_text)
        response2 = get_gemini_response(input_prompt2, resume_text, input_text)

        st.subheader("Evaluation Response:")
        st.write(response1)

        st.subheader("Percentage Match Response:")
        st.write(response2)
    else:
        st.error("Please upload the resume")
