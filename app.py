from dotenv import load_dotenv
import base64
import streamlit as st
import os
import io
import logging
from PIL import Image 
import pdf2image
import google.generativeai as genai
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GeminiATSClient:
    """Client for interacting with Gemini AI for ATS analysis"""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def get_response(self, input_text: str, pdf_content: List[Dict], prompt: str) -> str:
        """Get response from Gemini AI model"""
        try:
            response = self.model.generate_content([input_text, pdf_content[0], prompt])
            return response.text
        except Exception as e:
            logger.error(f"Error getting response from Gemini: {str(e)}")
            raise

class PDFProcessor:
    """Handles PDF processing and conversion"""
    
    @staticmethod
    def convert_pdf_to_images(uploaded_file) -> List[Image.Image]:
        """Convert PDF to list of images"""
        try:
            return pdf2image.convert_from_bytes(uploaded_file.read())
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
    
    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=85)
            return base64.b64encode(img_byte_arr.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            raise
    
    def process_uploaded_pdf(self, uploaded_file) -> Optional[List[Dict]]:
        """Process uploaded PDF file and return structured data"""
        if uploaded_file is None:
            return None
            
        try:
            images = self.convert_pdf_to_images(uploaded_file)
            if not images:
                raise ValueError("No images extracted from PDF")
                
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
    """Main ATS analysis orchestrator"""
    
    def __init__(self):
        self.gemini_client = GeminiATSClient()
        self.pdf_processor = PDFProcessor()
    
    def analyze_resume(self, job_description: str, uploaded_file, analysis_type: str) -> str:
        """Analyze resume based on the specified analysis type"""
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
        return """
        You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description. 
        Please provide a comprehensive summary covering:
        1. Candidate's technical exposure and knowledge
        2. Key strengths and weaknesses
        3. Overall suitability for the role
        4. Recommendations for improvement
        
        Be specific and provide actionable insights.
        """
    
    def _get_percentage_prompt(self) -> str:
        return """
        You are a skilled ATS (Applicant Tracking System) scanner with deep understanding of data science and ATS functionality. 
        Your task is to evaluate the resume against the provided job description and provide:
        
        1. MATCH PERCENTAGE: A precise percentage match between resume and job requirements
        2. MISSING KEYWORDS: Critical keywords and skills missing from the resume
        3. STRENGTHS: Keywords and skills that match well
        4. FINAL ASSESSMENT: Overall ATS compatibility and recommendations
        
        Format your response clearly with these sections.
        """
    
    def _get_evaluation_prompt(self) -> str:
        return """
        You are an experienced Technical Human Resource Manager. Your task is to provide a comprehensive evaluation of the resume against the job description.
        
        Please provide:
        1. ROLE ALIGNMENT: How well the candidate's profile aligns with the role
        2. TECHNICAL ASSESSMENT: Evaluation of technical skills and experience
        3. STRENGTHS: Key advantages and strong points
        4. AREAS FOR IMPROVEMENT: Gaps and weaknesses
        5. HIRING RECOMMENDATION: Final professional recommendation
        
        Be thorough and objective in your assessment.
        """

class StreamlitUI:
    """Handles Streamlit UI configuration and rendering"""
    
    def __init__(self):
        self.setup_page_config()
        self.apply_custom_styles()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="ATS Resume Analyzer Pro",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def apply_custom_styles(self):
        """Apply custom CSS styles for modern appearance"""
        st.markdown("""
        <style>
            .main {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333333;
            }
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }
            .stTextInput > div > div > input {
                background-color: #ffffff;
                color: #333333;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 10px;
            }
            .stTextArea textarea {
                background-color: #ffffff;
                color: #333333;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 10px;
            }
            .stButton>button {
                background: linear-gradient(45deg, #FF4B2B, #FF416C);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 16px;
                transition: all 0.3s ease;
                width: 100%;
                margin: 5px 0;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(255, 75, 43, 0.4);
            }
            .uploadedFile {
                background-color: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            }
            .success-message {
                background-color: #d4edda;
                color: #155724;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #c3e6cb;
                margin: 10px 0;
            }
            .error-message {
                background-color: #f8d7da;
                color: #721c24;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #f5c6cb;
                margin: 10px 0;
            }
            .analysis-section {
                background-color: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin: 10px 0;
            }
            .sidebar .sidebar-content {
                background: linear-gradient(180deg, #4facfe 0%, #00f2fe 100%);
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with instructions"""
        with st.sidebar:
            st.title("📋 Instructions")
            st.markdown("""
            ### How to use this ATS Analyzer:
            
            1. **Paste Job Description** in the main text area
            2. **Upload Your Resume** as a PDF file
            3. **Choose Analysis Type**:
               - **Resume Summary**: Detailed professional summary
               - **Percentage Match**: ATS compatibility score
               - **Complete Evaluation**: Comprehensive assessment
            
            ### Tips for Best Results:
            - Ensure job description is complete
            - Use standard PDF format for resume
            - Check file size (<10MB recommended)
            """)
            
            st.markdown("---")
            st.markdown("### 🛠️ About")
            st.markdown("""
            This ATS Resume Analyzer uses Google's Gemini AI to provide:
            - Intelligent resume analysis
            - ATS compatibility scoring
            - Professional hiring recommendations
            - Actionable improvement insights
            """)
    
    def render_main_interface(self, ats_analyzer: ATSAnalyzer):
        """Render the main interface"""
        st.title("🚀 ATS-Optimized Resume Analyzer")
        st.markdown("### Get AI-powered insights to optimize your resume for Applicant Tracking Systems")
        
        # Job description input
        with st.container():
            st.subheader("📝 Job Description")
            job_description = st.text_area(
                "Paste the job description here:",
                height=200,
                placeholder="Copy and paste the complete job description...",
                key="job_description"
            )
        
        # File upload
        with st.container():
            st.subheader("📄 Upload Resume")
            uploaded_file = st.file_uploader(
                "Choose your resume (PDF format):",
                type=["pdf"],
                help="Upload your resume in PDF format for analysis"
            )
            
            if uploaded_file is not None:
                st.markdown(
                    f'<div class="success-message">✅ PDF Uploaded Successfully: {uploaded_file.name}</div>', 
                    unsafe_allow_html=True
                )
        
        # Analysis buttons
        st.markdown("---")
        st.subheader("🔍 Analysis Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            summary_btn = st.button(
                "📋 Resume Summary", 
                key="summary",
                use_container_width=True
            )
        
        with col2:
            percentage_btn = st.button(
                "📊 Percentage Match", 
                key="percentage",
                use_container_width=True
            )
        
        with col3:
            evaluation_btn = st.button(
                "⭐ Complete Evaluation", 
                key="evaluation",
                use_container_width=True
            )
        
        # Handle analysis requests
        self.handle_analysis_requests(
            summary_btn, percentage_btn, evaluation_btn,
            job_description, uploaded_file, ats_analyzer
        )
    
    def handle_analysis_requests(self, summary_btn, percentage_btn, evaluation_btn, 
                               job_description, uploaded_file, ats_analyzer):
        """Handle analysis button clicks and display results"""
        
        if not job_description:
            st.warning("⚠️ Please enter a job description to proceed with analysis.")
            return
        
        if not uploaded_file:
            st.warning("⚠️ Please upload your resume PDF to proceed with analysis.")
            return
        
        try:
            if summary_btn:
                with st.spinner("🔍 Analyzing resume and generating summary..."):
                    response = ats_analyzer.analyze_resume(job_description, uploaded_file, "summary")
                    self.display_analysis_result("Resume Summary", response, "📋")
            
            elif percentage_btn:
                with st.spinner("📊 Calculating ATS compatibility percentage..."):
                    response = ats_analyzer.analyze_resume(job_description, uploaded_file, "percentage")
                    self.display_analysis_result("ATS Compatibility Analysis", response, "📊")
            
            elif evaluation_btn:
                with st.spinner("⭐ Conducting comprehensive evaluation..."):
                    response = ats_analyzer.analyze_resume(job_description, uploaded_file, "evaluation")
                    self.display_analysis_result("Complete Resume Evaluation", response, "⭐")
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")
    
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
    
    def add_download_button(self, content: str, title: str):
        """Add download button for analysis results"""
        timestamp = st.session_state.get('timestamp', 'analysis')
        filename = f"ATS_Analysis_{title.replace(' ', '_')}_{timestamp}.txt"
        
        st.download_button(
            label="💾 Download Analysis Results",
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
        
        # Render UI
        ui.render_sidebar()
        ui.render_main_interface(ats_analyzer)
        
    except Exception as e:
        st.error(f"🚨 Application initialization failed: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
