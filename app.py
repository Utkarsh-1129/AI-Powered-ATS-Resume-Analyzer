import streamlit as st
import os
import logging
import google.generativeai as genai
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureATSClient:
    """Secure ATS client with proper API key handling"""
    
    def __init__(self):
        # NEVER hardcode API keys - use Streamlit secrets or environment variables
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            st.error("""
            🔐 API Key Not Found
            
            Please add your Google API key to:
            - Streamlit Cloud: Settings → Secrets
            - Local: .streamlit/secrets.toml file
            """)
            st.stop()
        
        if api_key == "AIzaSyC1PtKcuJlU2G6CJ77_9marnwdvInBxIvQ":
            st.warning("⚠️ Please use a NEW API key - the one shown here was exposed publicly!")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.last_request_time = 0
        self.min_interval = 3
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get_response(self, input_text: str, resume_text: str, prompt: str) -> str:
        """Get response from Gemini AI"""
        self._rate_limit()
        
        try:
            full_prompt = f"""
            {prompt}
            
            JOB DESCRIPTION:
            {input_text}
            
            RESUME TEXT:
            {resume_text}
            """
            
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg:
                raise Exception("API quota exceeded. Please try again later.")
            elif "rate limit" in error_msg:
                raise Exception("Rate limit reached. Please wait a moment.")
            else:
                raise Exception(f"Analysis error: {str(e)}")

def main():
    st.set_page_config(
        page_title="Secure ATS Analyzer",
        page_icon="🔒",
        layout="wide"
    )
    
    st.title("🔒 Secure ATS Resume Analyzer")
    st.markdown("AI-powered resume analysis with secure API handling")
    
    # Initialize client
    try:
        client = SecureATSClient()
        st.success("✅ API configured successfully")
    except Exception as e:
        st.error(f"❌ Configuration failed: {str(e)}")
        return
    
    # Input sections
    job_description = st.text_area(
        "📝 Job Description",
        height=150,
        placeholder="Paste the job description here..."
    )
    
    resume_text = st.text_area(
        "📄 Resume Text", 
        height=250,
        placeholder="Paste your resume text here..."
    )
    
    # Analysis buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Quick Summary", use_container_width=True):
            if job_description and resume_text:
                with st.spinner("Analyzing..."):
                    try:
                        prompt = "Provide a concise summary of how well the resume matches the job description."
                        response = client.get_response(job_description, resume_text, prompt)
                        st.subheader("Summary")
                        st.write(response)
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("Please enter both job description and resume text")
    
    with col2:
        if st.button("📊 Match Score", use_container_width=True):
            if job_description and resume_text:
                with st.spinner("Calculating score..."):
                    try:
                        prompt = "Provide a percentage match score and list matching/missing keywords."
                        response = client.get_response(job_description, resume_text, prompt)
                        st.subheader("ATS Score")
                        st.write(response)
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("Please enter both job description and resume text")
    
    with col3:
        if st.button("⭐ Full Evaluation", use_container_width=True):
            if job_description and resume_text:
                with st.spinner("Evaluating..."):
                    try:
                        prompt = "Provide a comprehensive evaluation with strengths, weaknesses, and recommendations."
                        response = client.get_response(job_description, resume_text, prompt)
                        st.subheader("Full Evaluation")
                        st.write(response)
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("Please enter both job description and resume text")

if __name__ == "__main__":
    main()
