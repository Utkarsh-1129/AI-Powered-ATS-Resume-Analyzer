import streamlit as st
import os
import base64
import io
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import time
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SimpleATSClient:
    """Simplified ATS client that avoids complex dependencies"""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("🚨 GOOGLE_API_KEY not found. Please add it to your environment variables.")
            st.stop()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.last_request_time = 0
        self.min_interval = 3  # Increased interval for free tier
    
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

class TextProcessor:
    """Handles text processing without PDF dependencies"""
    
    @staticmethod
    def read_uploaded_file(uploaded_file):
        """Read uploaded file as text"""
        try:
            if uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                # For PDF files, we'll use a text-based approach
                # In a real scenario, you might want to use a simpler PDF library
                return f"PDF file uploaded: {uploaded_file.name}. Please paste your resume text below or use a text-based resume."
            else:
                return f"File uploaded: {uploaded_file.name}. Please paste your resume text in the text area below."
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

class ATSAnalyzer:
    """Main ATS analysis orchestrator"""
    
    def __init__(self):
        self.gemini_client = SimpleATSClient()
        self.text_processor = TextProcessor()
        self.analysis_count = 0
        self.max_analyses_per_session = 15  # Increased for free tier
    
    def analyze_resume(self, job_description: str, resume_text: str, analysis_type: str) -> str:
        """Analyze resume with usage limits"""
        self.analysis_count += 1
        if self.analysis_count > self.max_analyses_per_session:
            raise Exception(f"Session limit reached. Maximum {self.max_analyses_per_session} analyses per session.")
        
        prompts = {
            "summary": self._get_summary_prompt(),
            "percentage": self._get_percentage_prompt(),
            "evaluation": self._get_evaluation_prompt()
        }
        
        if analysis_type not in prompts:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
        
        try:
            response = self.gemini_client.get_response(
                job_description, 
                resume_text, 
                prompts[analysis_type]
            )
            return response
        except Exception as e:
            logger.error(f"Error in resume analysis: {str(e)}")
            raise
    
    def _get_summary_prompt(self) -> str:
        return """You are an experienced Technical HR Manager. Analyze the resume against the job description and provide:

1. TECHNICAL SKILLS MATCH: Key matching technical skills
2. EXPERIENCE ALIGNMENT: How well experience matches requirements  
3. STRENGTHS: Candidate's main advantages
4. GAPS: Missing qualifications or experience
5. OVERALL ASSESSMENT: Brief summary of fit

Be concise and practical."""

    def _get_percentage_prompt(self) -> str:
        return """You are an ATS expert. Evaluate the resume against the job description and provide:

MATCH PERCENTAGE: XX% (provide a realistic percentage)
MATCHING KEYWORDS: [list 5-7 key matching terms]
MISSING KEYWORDS: [list 5-7 important missing terms]
RECOMMENDATIONS: [3-4 actionable suggestions]

Focus on ATS compatibility and keyword optimization."""

    def _get_evaluation_prompt(self) -> str:
        return """As a Senior HR Professional, provide a comprehensive evaluation:

ROLE SUITABILITY: How well the candidate fits the role
TECHNICAL COMPETENCY: Assessment of technical skills
EXPERIENCE RELEVANCE: Relevance of past experience
CULTURE/TEAM FIT: Potential fit with team culture
HIRING RECOMMENDATION: Strong recommend / Recommend / Not recommended
KEY IMPROVEMENTS: Specific areas for resume improvement

Provide honest, professional assessment."""

class StreamlitUI:
    """Handles Streamlit UI configuration"""
    
    def __init__(self):
        self.setup_page_config()
        self.apply_custom_styles()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Free ATS Resume Analyzer",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def apply_custom_styles(self):
        """Apply custom CSS styles"""
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
            .success-message {
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
                font-weight: 600;
            }
            .warning-message {
                background: linear-gradient(45deg, #FF9800, #F57C00);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
                font-weight: 600;
            }
            .error-message {
                background: linear-gradient(45deg, #f44336, #d32f2f);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
                font-weight: 600;
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
            .sidebar .sidebar-content {
                background: linear-gradient(180deg, #4facfe 0%, #00f2fe 100%);
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self, ats_analyzer: ATSAnalyzer):
        """Render the sidebar"""
        with st.sidebar:
            st.title("🎯 Free ATS Analyzer")
            
            # Usage counter
            remaining = max(0, ats_analyzer.max_analyses_per_session - ats_analyzer.analysis_count)
            st.markdown(f"""
            <div class="usage-counter">
                <h3>📊 Session Usage</h3>
                <h2>{remaining} / {ats_analyzer.max_analyses_per_session}</h2>
                <p>Analyses Remaining</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💡 How To Use")
            st.markdown("""
            1. **Paste Job Description** in the main area
            2. **Paste Your Resume** text in the resume area
            3. **Choose Analysis Type**:
               - **Quick Summary**: Overview of fit
               - **ATS Score**: Compatibility percentage
               - **Full Evaluation**: Detailed assessment
            
            *No file uploads required!*
            """)
            
            st.markdown("### 🚀 Free Tier Benefits")
            st.markdown("""
            - ✅ 15 analyses per session
            - ✅ No file size limits  
            - ✅ No dependencies
            - ✅ Fast processing
            - ✅ Mobile friendly
            """)
            
            st.markdown("---")
            st.markdown("""
            **Pro Tip:** Copy-paste your resume text from:
            - LinkedIn profile
            - Word/PDF document  
            - Google Docs
            - Any text editor
            """)
    
    def render_main_interface(self, ats_analyzer: ATSAnalyzer):
        """Render the main interface"""
        st.title("🎯 Free ATS Resume Analyzer")
        st.markdown("### Get instant AI-powered resume feedback - No dependencies required!")
        
        # Free tier notice
        st.markdown("""
        <div style="background: linear-gradient(45deg, #FF9800, #F57C00); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <strong>🎁 Text-Based Version:</strong> Simply paste your job description and resume text below - no file uploads needed!
        </div>
        """, unsafe_allow_html=True)
        
        # Job description input
        with st.container():
            st.subheader("📝 Job Description")
            job_description = st.text_area(
                "Paste the complete job description:",
                height=200,
                placeholder="Copy and paste the entire job description here...",
                key="job_description",
                help="Include requirements, responsibilities, and qualifications"
            )
        
        # Resume text input
        with st.container():
            st.subheader("📄 Resume Text")
            resume_text = st.text_area(
                "Paste your resume text:",
                height=300,
                placeholder="Copy and paste your resume text here...\n\nYou can get this from:\n- LinkedIn 'Save to PDF'\n- Word/Google Docs document\n- Any text-based resume",
                key="resume_text",
                help="Include your experience, skills, education, and projects"
            )
            
            # Optional file upload for reference (but we'll use text)
            with st.expander("💾 Optional: Upload File for Reference"):
                uploaded_file = st.file_uploader(
                    "Upload any file (we'll extract the name only):",
                    type=["txt", "pdf", "docx"],
                    help="This is just for reference - analysis uses pasted text"
                )
                if uploaded_file is not None:
                    st.info(f"📎 File reference: {uploaded_file.name}")
        
        # Analysis buttons
        st.markdown("---")
        st.subheader("🔍 Choose Analysis Type")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            summary_btn = st.button(
                "📋 Quick Summary", 
                key="summary",
                use_container_width=True,
                help="Fast overview of resume match"
            )
        
        with col2:
            percentage_btn = st.button(
                "📊 ATS Score", 
                key="percentage",
                use_container_width=True,
                help="Compatibility percentage and keywords"
            )
        
        with col3:
            evaluation_btn = st.button(
                "⭐ Full Evaluation", 
                key="evaluation",
                use_container_width=True,
                help="Comprehensive assessment"
            )
        
        # Handle analysis requests
        self.handle_analysis_requests(
            summary_btn, percentage_btn, evaluation_btn,
            job_description, resume_text, ats_analyzer
        )
    
    def handle_analysis_requests(self, summary_btn, percentage_btn, evaluation_btn, 
                               job_description, resume_text, ats_analyzer):
        """Handle analysis button clicks"""
        
        # Check prerequisites
        if not job_description or not job_description.strip():
            st.warning("📝 Please enter a job description to start analysis.")
            return
        
        if not resume_text or not resume_text.strip():
            st.warning("📄 Please paste your resume text to continue.")
            return
        
        # Check usage limits
        remaining = ats_analyzer.max_analyses_per_session - ats_analyzer.analysis_count
        if remaining <= 0:
            st.error("""
            ❌ Session limit reached! 
            
            You've used all your free analyses for this session. 
            Please refresh the page to start a new session.
            """)
            return
        
        try:
            if summary_btn:
                with st.spinner(f"🔍 Analyzing resume fit... ({remaining} analyses left)"):
                    response = ats_analyzer.analyze_resume(job_description, resume_text, "summary")
                    self.display_analysis_result("Quick Summary", response, "📋")
            
            elif percentage_btn:
                with st.spinner(f"📊 Calculating ATS compatibility... ({remaining} analyses left)"):
                    response = ats_analyzer.analyze_resume(job_description, resume_text, "percentage")
                    self.display_analysis_result("ATS Compatibility Score", response, "📊")
            
            elif evaluation_btn:
                with st.spinner(f"⭐ Conducting comprehensive evaluation... ({remaining} analyses left)"):
                    response = ats_analyzer.analyze_resume(job_description, resume_text, "evaluation")
                    self.display_analysis_result("Complete Evaluation", response, "⭐")
        
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                st.error("""
                💳 API Quota Exceeded!
                
                You've reached the free tier limit. 
                Please try again tomorrow.
                """)
            elif "rate limit" in error_msg.lower():
                st.warning("""
                ⏳ Rate limit reached!
                
                Please wait a few seconds between requests.
                """)
            else:
                st.error(f"❌ Analysis failed: {error_msg}")
            
            logger.error(f"Analysis error: {error_msg}")
    
    def display_analysis_result(self, title: str, content: str, icon: str):
        """Display analysis results"""
        st.markdown("---")
        st.markdown(f"### {icon} {title}")
        
        with st.container():
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add download button
            self.add_download_button(content, title)
    
    def add_download_button(self, content: str, title: str):
        """Add download button for results"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"ATS_Analysis_{title.replace(' ', '_')}_{timestamp}.txt"
        
        st.download_button(
            label="💾 Download Results",
            data=content,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )

def main():
    """Main application entry point"""
    try:
        # Initialize components
        ui = StreamlitUI()
        ats_analyzer = ATSAnalyzer()
        
        # Store analyzer in session state
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = ats_analyzer
        
        # Render UI
        ui.render_sidebar(ats_analyzer)
        ui.render_main_interface(ats_analyzer)
        
    except Exception as e:
        st.error(f"🚨 Application error: {str(e)}")
        logger.error(f"Application initialization failed: {str(e)}")

if __name__ == "__main__":
    main()
