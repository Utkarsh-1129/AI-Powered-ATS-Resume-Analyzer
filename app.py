from dotenv import load_dotenv
import base64
import streamlit as st
import os
import io
from PIL import Image
import pdf2image
import google.generativeai as genai

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("⚠️ Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
else:
    genai.configure(api_key=API_KEY)

# ---------------------- Utility Functions ----------------------

def get_gemini_response(system_prompt, pdf_content, job_desc):
    """Send resume + job description to Gemini 2.5 Flash model with fallback."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([system_prompt, pdf_content[0], job_desc])
        return response.text
    except Exception as e:
        # Fallback to Pro model if Flash 2.5 is unavailable
        st.warning("⚠️ Falling back to gemini-2.0-pro due to: {}".format(e))
        model = genai.GenerativeModel("gemini-2.0-pro")
        response = model.generate_content([system_prompt, pdf_content[0], job_desc])
        return response.text

def input_pdf_setup(uploaded_file):
    """Convert PDF into base64-encoded images for Gemini."""
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")

    images = pdf2image.convert_from_bytes(uploaded_file.read())
    first_page = images[0]  # Only first page for efficiency

    img_byte_arr = io.BytesIO()
    first_page.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    pdf_parts = [
        {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }
    ]
    return pdf_parts

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="AI Powered Resume Analyzer", page_icon="📄", layout="centered")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    *{
        
    }
    /* MAIN Styling */
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f0f4f8, #d);
    }

    .stApp {
            background: linear-gradient(135deg, #f0f4, #406078);


    }

    /* Header */
    .stHeader {
        font-weight: 600;
        font-size: 1.5rem;
        color: #1a202c;
        text-align: center;
        padding-bottom: 20px;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #007BFF;
        color: white;
        font-weight: 500;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        text-align: center;
        
    }

    .stButton>button:hover {
        background-color: #132743;
        color: yellow;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background-color: white;
        color: #495057;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 10px;
        text-align: center;    
    }

    /* Columns and Sections */
    [data-testid="stHorizontalBlock"] > div {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Subheader */
    .stSubheader {
        color: #1a202c;
        font-weight: 1000;
        margin-bottom: 15px;
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


#prompt to assign task to Gen AI
input_prompt1 = """
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
Please share your feedback on candidate exposure and Techical knowledge improvement according to the job requirement. 
Highlight the flaws and weaknesses of the applicant.
"""

input_prompt2 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to strictly evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First, the output should come as percentage and then keywords missing and last final  and don't show any leniency.
"""
input_prompt3 ="""
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

#submit actions
if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.error("Please upload the resume")

elif submit2:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt2, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.error("Please upload the resume")

elif submit3:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response1 = get_gemini_response(input_prompt1, pdf_content, input_text)
        response2 = get_gemini_response(input_prompt2, pdf_content, input_text)

        st.subheader("Evaluation Response:")
        st.write(response1)

        st.subheader("Percentage Match Response:")
        st.write(response2)
    else:
        st.error("Please upload the resume.")
