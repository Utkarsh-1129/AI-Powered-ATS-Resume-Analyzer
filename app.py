from dotenv import load_dotenv
import base64
import streamlit as st
import os
import io
import logging
from PIL import Image 
import pdf2image
import google.generativeai as genai
import time
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FreeTierATSClient:
    """Client optimized for free tier usage with rate limiting"""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("🚨 GOOGLE_API_KEY not found. Please add it to your .env file")
            st.stop()
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.last_request_time = 0
        self.min_interval = 2  # Minimum seconds between requests
        
    def _rate_limit(self):
        """Implement basic rate limiting for free tier"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get_response(self, input_text: str, pdf_content: List[Dict], prompt: str) -> str:
        """Get response with rate limiting and error handling"""
        self._rate_limit()
        
        try:
            response = self.model.generate_content([input_text, pdf_content[0], prompt])
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate limit" in error_msg:
                raise Exception("⚠️ API quota exceeded. Please try again later or check your Google AI API usage.")
            elif "safety" in error_msg:
                raise Exception("🔒 Content was blocked for safety reasons. Please modify your input.")
            else:
                raise Exception(f"❌ API error: {str(e)}")

class PDFProcessor:
    """Handles PDF processing with free tier optimizations"""
    
    @staticmethod
    def validate_pdf_size(uploaded_file) -> bool:
        """Validate PDF size for free tier constraints"""
        max_size_mb = 5
        if uploaded_file.size > max_size_mb * 1024 * 1024:
            raise ValueError(f"File size too large. Maximum {max_size_mb}MB allowed.")
        return True
    
    @staticmethod
    def convert_pdf_to_images(uploaded_file) -> List[Image.Image]:
        """Convert PDF to images with optimization"""
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            images = pdf2image.convert_from_bytes(
                uploaded_file.read(),
                dpi=150,  # Lower DPI for faster processing
                first_page=1,
                last_page=2  # Process only first 2 pages for free tier
            )
            return images
        except Exception as e:
            raise Exception(f"Error converting PDF: {str(e)}")
    
    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """Convert PIL image to base64 with compression"""
        try:
            img_byte_arr = io.BytesIO()
            # Compress image for free tier
            image = image.convert('RGB')
            image.save(img_byte_arr, format='JPEG', quality=70, optimize=True)
            return base64.b64encode(img_byte_arr.getvalue()).decode()
        except Exception as e:
            raise Exception(f"Error converting image: {str(e)}")
    
    def process_uploaded_pdf(self, uploaded_file) -> Optional[List[Dict]]:
        """Process uploaded PDF with free tier optimizations"""
        if uploaded_file is None:
            return None
            
        try:
            # Validate file size
            self.validate_pdf_size(uploaded_file)
            
            # Convert PDF to images
            images = self.convert_pdf_to_images(uploaded_file)
            if not images:
                raise ValueError("No pages could be extracted from PDF")
                
            # Use only first page for free tier
            first_page = images[0]
            pdf_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": self.image_to_base64(first_page)
                }
            ]
            return pdf_parts
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

class ATSAnalyzer:
    """Main ATS analysis orchestrator optimized for free tier"""
    
    def __init__(self):
        self.gemini_client = FreeTierATSClient()
        self.pdf_processor = PDFProcessor()
        self.analysis_count = 0
        self.max_analyses_per_session = 10
    
    def analyze_resume(self, job_description: str, uploaded_file, analysis_type: str) -> str:
        """Analyze resume with usage limits"""
        # Check usage limits
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
            pdf_content = self.pdf_processor.process_uploaded_pdf(uploaded_file)
            if not pdf_content:
                raise ValueError("No PDF content available for analysis")
                
            response = self.gemini_client.get_response(
                prompts[analysis_type], 
                pdf_content, 
                job_description
            )
            return response
        except Exception as e:
            logger.error(f"Error in resume analysis: {str(e)}")
            raise
    
    def _get_summary_prompt(self) -> str:
        return """Provide a concise resume summary focusing on:
1. Technical skills match
2. Key strengths
3. Major gaps
4. Overall impression

Keep response under 300 words."""

    def _get_percentage_prompt(self) -> str:
        return """Evaluate ATS compatibility and provide:
MATCH SCORE: XX%
MISSING KEYWORDS: [list]
STRENGTHS: [list]
QUICK ASSESSMENT: [brief]

Be concise and focused."""

    def _get_evaluation_prompt(self) -> str:
        return """Provide a focused evaluation:
1. Role Alignment: [brief]
2. Technical Fit: [key points]
3. Recommendations: [actionable]

Keep response structured but concise."""

class StreamlitUI:
    """Handles Streamlit UI with free tier optimizations"""
    
    def __init__(self):
        self.setup_page_config()
        self.apply_free_tier_styles()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Free ATS Resume Analyzer",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def apply_free_tier_styles(self):
        """Apply optimized CSS styles for free tier"""
        st.markdown("""
        <style>
            .main {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stTextInput > div > div > input {
                background-color: #ffffff;
                color: #333333;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
            }
            .stTextArea textarea {
                background-color: #ffffff;
                color: #333333;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
            }
            .stButton>button {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                border-radius: 20px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                transition: all 0.3s ease;
                width: 100%;
                margin: 8px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stButton>button:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .stButton>button:disabled {
                background: #cccccc;
                transform: none;
                box-shadow: none;
            }
            .uploadedFile {
                background-color: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
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
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin: 15px 0;
                border-left: 5px solid #667eea;
            }
            .usage-counter {
                background: white;
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                margin: 10px 0;
                border: 2px solid #667eea;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self, ats_analyzer: ATSAnalyzer):
        """Render the sidebar with free tier information"""
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
            
            st.markdown("### 💡 Free Tier Tips")
            st.markdown("""
            **To optimize your free usage:**
            - Keep job descriptions focused
            - Use standard PDF resumes
            - Files under 5MB work best
            - Process only first 2 pages
            
            **Limitations:**
            - Max 10 analyses per session
            - Rate limited requests
            - First page analysis only
            """)
            
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            1. Paste job description
            2. Upload PDF resume
            3. Choose analysis type
            4. Get instant feedback
            """)
            
            st.markdown("---")
            st.markdown("""
            *Built for free tier usage*  
            *Uses Google Gemini AI*
            """)
    
    def render_main_interface(self, ats_analyzer: ATSAnalyzer):
        """Render the main interface"""
        st.title("🎯 Free ATS Resume Analyzer")
        st.markdown("### Get AI-powered resume insights without costs")
        
        # Free tier notice
        st.markdown("""
        <div style="background: linear-gradient(45deg, #FF9800, #F57C00); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <strong>🎁 Free Tier Notice:</strong> This version is optimized for free usage with limited analyses per session.
        </div>
        """, unsafe_allow_html=True)
        
        # Job description input
        with st.container():
            st.subheader("📝 Job Description")
            job_description = st.text_area(
                "Paste the job description here:",
                height=150,
                placeholder="Copy and paste the job description... (Keep it concise for free tier)",
                key="job_description",
                help="For free tier, keep job descriptions under 2000 characters"
            )
            
            if job_description and len(job_description) > 2000:
                st.warning("⚠️ Long job description may exceed free tier limits. Consider shortening.")
        
        # File upload
        with st.container():
            st.subheader("📄 Upload Resume (PDF)")
            uploaded_file = st.file_uploader(
                "Choose your resume:",
                type=["pdf"],
                help="Max 5MB, first 2 pages processed"
            )
            
            if uploaded_file is not None:
                try:
                    # Validate file
                    if uploaded_file.size > 5 * 1024 * 1024:
                        st.error("❌ File too large. Maximum 5MB allowed.")
                    else:
                        st.markdown(
                            f'<div class="success-message">✅ PDF Uploaded: {uploaded_file.name} ({uploaded_file.size // 1024}KB)</div>', 
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.error(f"❌ Error with uploaded file: {str(e)}")
        
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
            job_description, uploaded_file, ats_analyzer
        )
    
    def handle_analysis_requests(self, summary_btn, percentage_btn, evaluation_btn, 
                               job_description, uploaded_file, ats_analyzer):
        """Handle analysis button clicks with free tier limits"""
        
        # Check prerequisites
        if not job_description:
            st.warning("📝 Please enter a job description to start analysis.")
            return
        
        if not uploaded_file:
            st.warning("📄 Please upload your resume PDF to continue.")
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
                with st.spinner(f"🔍 Generating summary... ({remaining} left)"):
                    response = ats_analyzer.analyze_resume(job_description, uploaded_file, "summary")
                    self.display_analysis_result("Quick Summary", response, "📋")
            
            elif percentage_btn:
                with st.spinner(f"📊 Calculating ATS score... ({remaining} left)"):
                    response = ats_analyzer.analyze_resume(job_description, uploaded_file, "percentage")
                    self.display_analysis_result("ATS Compatibility Score", response, "📊")
            
            elif evaluation_btn:
                with st.spinner(f"⭐ Evaluating resume... ({remaining} left)"):
                    response = ats_analyzer.analyze_resume(job_description, uploaded_file, "evaluation")
                    self.display_analysis_result("Complete Evaluation", response, "⭐")
        
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower():
                st.error("""
                💳 API Quota Exceeded!
                
                You've reached the free tier limit for today. 
                Please try again tomorrow or check your Google AI API dashboard.
                """)
            elif "rate limit" in error_msg.lower():
                st.warning("""
                ⏳ Rate limit reached!
                
                Please wait a few seconds between requests.
                The free tier has limited requests per minute.
                """)
            else:
                st.error(f"❌ Analysis failed: {error_msg}")
            
            logger.error(f"Analysis error: {error_msg}")
    
    def display_analysis_result(self, title: str, content: str, icon: str):
        """Display analysis results in a formatted way"""
        st.markdown("---")
        st.markdown(f"### {icon} {title}")
        
        with st.container():
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add download button for results
            self.add_download_button(content, title)
            
            # Show remaining analyses
            remaining = st.session_state.get('analyzer', None)
            if remaining:
                st.info(f"🔄 You have {remaining} analyses remaining this session.")
    
    def add_download_button(self, content: str, title: str):
        """Add download button for analysis results"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"Free_ATS_Analysis_{title.replace(' ', '_')}_{timestamp}.txt"
        
        st.download_button(
            label="💾 Download Results",
            data=content,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )

def main():
    """Main application entry point optimized for free tier"""
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
