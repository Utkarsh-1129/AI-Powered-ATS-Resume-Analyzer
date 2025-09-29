import streamlit as st
import os
import logging
import google.generativeai as genai
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureATSClient:
    """Secure ATS client without Pillow dependency"""
    
    def __init__(self):
        # Use Streamlit secrets or environment variables
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            st.error("""
            🔐 API Key Not Found
            
            Please add your Google API key to:
            - Streamlit Cloud: Settings → Secrets
            - Local: .streamlit/secrets.toml file
            - Environment variable: GOOGLE_API_KEY
            """)
            st.stop()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.last_request_time = 0
        self.min_interval = 3
        self.analysis_count = 0
        self.max_analyses = 20
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_request_time = time.time()
    
    def analyze_resume(self, job_description: str, resume_text: str, analysis_type: str) -> str:
        """Analyze resume with usage limits"""
        self.analysis_count += 1
        if self.analysis_count > self.max_analyses:
            raise Exception(f"Session limit reached. Maximum {self.max_analyses} analyses per session.")
        
        self._rate_limit()
        
        prompts = {
            "summary": """
            Provide a concise resume summary focusing on:
            1. Technical skills match
            2. Key strengths
            3. Major gaps  
            4. Overall impression
            Keep response under 300 words.
            """,
            "percentage": """
            Evaluate ATS compatibility and provide:
            MATCH SCORE: XX%
            MISSING KEYWORDS: [list]
            STRENGTHS: [list]
            QUICK ASSESSMENT: [brief]
            Be concise and focused.
            """,
            "evaluation": """
            Provide a comprehensive evaluation:
            1. Role Alignment
            2. Technical Fit  
            3. Experience Relevance
            4. Hiring Recommendation
            5. Key Improvements
            Provide honest, professional assessment.
            """
        }
        
        if analysis_type not in prompts:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
        
        try:
            full_prompt = f"""
            {prompts[analysis_type]}
            
            JOB DESCRIPTION:
            {job_description}
            
            RESUME TEXT:
            {resume_text}
            
            Please provide your analysis based on the above information.
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
    # Page configuration
    st.set_page_config(
        page_title="ATS Resume Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            color: #333333;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            font-size: 14px;
        }
        .stButton>button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 15px 25px;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
            width: 100%;
            margin: 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .analysis-section {
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 15px 0;
            border-left: 5px solid #667eea;
        }
        .usage-counter {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 10px 0;
            border: 2px solid #667eea;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize client
    try:
        client = SecureATSClient()
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 ATS Analyzer")
        
        remaining = max(0, client.max_analyses - client.analysis_count)
        st.markdown(f"""
        <div class="usage-counter">
            <h3>📊 Session Usage</h3>
            <h2>{remaining} / {client.max_analyses}</h2>
            <p>Analyses Remaining</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 How To Use")
        st.markdown("""
        1. **Paste Job Description** in the main area
        2. **Paste Your Resume** text in the resume area  
        3. **Choose Analysis Type**
        
        *No file uploads required!*
        """)
    
    # Main interface
    st.title("🎯 ATS Resume Analyzer")
    st.markdown("### Get AI-powered resume feedback in seconds!")
    
    # Input sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Paste the job description:",
            height=250,
            placeholder="Copy and paste the complete job description here...",
            label_visibility="collapsed"
        )
    
    with col2:
        st.subheader("📄 Resume Text")
        resume_text = st.text_area(
            "Paste your resume:",
            height=250,
            placeholder="Copy and paste your resume text here...\n\nYou can get this from:\n- LinkedIn profile\n- Word/Google Docs\n- Any text editor",
            label_visibility="collapsed"
        )
    
    # Analysis buttons
    st.markdown("---")
    st.subheader("🔍 Choose Analysis Type")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        summary_btn = st.button("📋 Quick Summary", use_container_width=True)
    with col2:
        percentage_btn = st.button("📊 ATS Score", use_container_width=True)
    with col3:
        evaluation_btn = st.button("⭐ Full Evaluation", use_container_width=True)
    
    # Handle analysis requests
    if not job_description or not resume_text:
        st.info("👆 Please enter both job description and resume text to get started.")
    else:
        try:
            if summary_btn:
                with st.spinner("🔍 Analyzing resume fit..."):
                    response = client.analyze_resume(job_description, resume_text, "summary")
                    st.markdown("### 📋 Quick Summary")
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    st.markdown(response)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            elif percentage_btn:
                with st.spinner("📊 Calculating ATS compatibility..."):
                    response = client.analyze_resume(job_description, resume_text, "percentage")
                    st.markdown("### 📊 ATS Compatibility Score")
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    st.markdown(response)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            elif evaluation_btn:
                with st.spinner("⭐ Conducting comprehensive evaluation..."):
                    response = client.analyze_resume(job_description, resume_text, "evaluation")
                    st.markdown("### ⭐ Complete Evaluation")
                    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                    st.markdown(response)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ {str(e)}")

if __name__ == "__main__":
    main()
