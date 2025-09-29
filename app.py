from dotenv import load_dotenv
import base64
import streamlit as st
import os
import io
from PIL import Image 
import pdf2image
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Streamlit App
st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄", layout="wide")

# Apply dark theme
st.markdown("""
<style>
    .stApp {
        background-color: grey;
        color: #1E1E1E;
    }
    .stTextInput > div > div > input {
        background-color: #2B2B2B;
        color: #FFFFFF;
    }
    .stTextArea textarea {
        background-color: #FFFFFF;
        color: black;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .uploadedFile {
        background-color: #2B2B2B;
    }
    .css-1cpxqw2 {
        color: #FFFFFF;
    }
    .css-5uatcg {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .css-5uatcg:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

st.header("ATS-Optimised Tracking System")
input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload your resume here (PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF Uploaded Successfully")

col1, col2, col3 = st.columns(3)

with col1:
    submit1 = st.button("Resume Summary")

with col2:
    submit2 = st.button("Percentage match")

with col3:
    submit3 = st.button("Evaluate Resume")

input_prompt1 = """
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
Please share your Summary on candidate exposure and Techical knowledge. 
Highlight the strengths and weaknesses of the applicant.
"""

input_prompt2 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. Give me the percentage of match if the resume matches
the job description. First, the output should come as percentage and then keywords missing and last final thoughts.
"""
input_prompt3 ="""
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""


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
