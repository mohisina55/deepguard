import streamlit as st
import os
import time
from detect_video import analyze_video_ultra
import tempfile
import base64
from PIL import Image

# Configure page
st.set_page_config(
    page_title="DeepGuard - Video Authenticity Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            font-family: 'Inter', sans-serif;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #3b82f6, #1d4ed8);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            padding: 12px 28px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            font-family: 'Inter', sans-serif;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        }
        
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin: 0.5rem 0;
        }
        
        .hero-title {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            font-family: 'Inter', sans-serif;
        }
        
        .hero-subtitle {
            text-align: center;
            color: #475569;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        .sidebar .stMarkdown {
            font-family: 'Inter', sans-serif;
        }
        
        .upload-section {
            background: white;
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #3b82f6;
            background: #f8fafc;
        }
        
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            border: 1px solid #e2e8f0;
            margin: 1rem 0;
        }
        
        .warning-card {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 1px solid #f59e0b;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .success-card {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border: 1px solid #10b981;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .error-card {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border: 1px solid #ef4444;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .stFileUploader > div > div {
            background: transparent !important;
        }
        
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .how-it-works-step {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .stVideo {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #3b82f6; font-size: 2rem; margin-bottom: 0.5rem;">🛡️ DeepGuard</h1>
        <p style="color: #475569; font-size: 0.9rem; margin-bottom: 1rem;">Video Authenticity Verification</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 How It Works
    """)
    
    st.markdown("""
    <div class="how-it-works-step">
        <strong style="color: #1e293b;">1. 📁 Upload Video</strong><br>
        <span style="color: #64748b; font-size: 0.9rem;">Support for MP4, MOV, AVI formats</span>
    </div>
    
    <div class="how-it-works-step">
        <strong style="color: #1e293b;">2. 🔍 Video Analysis</strong><br>
        <span style="color: #64748b; font-size: 0.9rem;">Advanced detection scans every frame</span>
    </div>
    
    <div class="how-it-works-step">
        <strong style="color: #1e293b;">3. 📊 Detailed Report</strong><br>
        <span style="color: #64748b; font-size: 0.9rem;">Comprehensive authenticity assessment</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔍 Detection Features
    """)
    
    st.markdown("""
    - **Facial Inconsistencies**
    - **Movement Patterns** 
    - **Compression Analysis**
    - **Texture Verification**
    - **Frame Consistency**
    """)

# Enhanced Main Content
st.markdown('<h1 class="hero-title">DeepGuard</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Advanced Video Authenticity Detection System</p>', unsafe_allow_html=True)

# File uploader with enhanced styling
st.markdown("""
<div class="upload-section">
    <div style="font-size: 3rem; margin-bottom: 1rem;">📹</div>
    <h3 style="color: #1e293b; margin-bottom: 0.5rem;">Upload Video for Analysis</h3>
    <p style="color: #475569; margin-bottom: 1rem;">Drag and drop your video file or click to browse</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a video file", 
    type=["mp4", "mov", "avi"],
    accept_multiple_files=False,
    label_visibility="collapsed"
)

# Temporary file handling
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Enhanced Analysis function
def analyze_video(video_path):
    try:
        st.markdown("""
        <div class="result-card">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">🔍 Analysis in Progress</h3>
            <p style="color: #475569;">Examining your video frame by frame...</p>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progress with enhanced messaging
        messages = [
            "Initializing detection systems...",
            "Extracting video frames...", 
            "Analyzing facial features...",
            "Checking movement patterns...",
            "Computing texture metrics...",
            "Evaluating frame consistency...",
            "Generating final report..."
        ]
        
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
            if i % 15 == 0:
                msg_idx = min(i // 15, len(messages) - 1)
                status_text.markdown(f"**{messages[msg_idx]}**")
        
        # Actual analysis
        result = analyze_video_ultra(video_path)
        progress_bar.empty()
        status_text.empty()
        
        return result
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

# Enhanced video display
def display_video(video_path):
    st.markdown("""
    <div class="result-card">
        <h3 style="color: #1e293b; margin-bottom: 1rem;">📺 Video Preview</h3>
                <style>
    .stVideo {
        width: 70% !important;
        margin:  auto;
    }
    </style>
    </div>
    """, unsafe_allow_html=True)
    
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    

# Enhanced result visualization
def show_results(result):
    st.markdown("---")
    
    # Results header
    st.markdown("""
    <div class="result-card">
        <h2 style="color: #1e293b; text-align: center; margin-bottom: 2rem;">📊 Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns with better spacing
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Confidence meter with enhanced styling
        confidence = result['confidence'] * 100
        
        if confidence >= 80:
            confidence_color = "#10b981"
            confidence_bg = "#d1fae5"
        elif confidence >= 60:
            confidence_color = "#f59e0b"  
            confidence_bg = "#fef3c7"
        else:
            confidence_color = "#ef4444"
            confidence_bg = "#fee2e2"
        
        st.markdown(f"""
        <div style="background: {confidence_bg}; padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem; border: 1px solid {confidence_color}33;">
            <p style="color: #475569; margin: 0; font-size: 1rem; font-weight: 500;">Confidence Score</p>
            <p style="color: {confidence_color}; font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;">{confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Verdict badge with enhanced styling
        if result['prediction'] == "Fake":
            st.markdown("""
            <div style="background: #fee2e2; border: 1px solid #ef444433; border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">❌</div>
                <p style="font-weight: 600; color: #dc2626; margin: 0; font-size: 1.1rem;">Altered Content Detected</p>
                <p style="color: #475569; font-size: 0.9rem; margin: 0.5rem 0 0 0;">High probability of manipulation</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #d1fae5; border: 1px solid #10b98133; border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">✅</div>
                <p style="font-weight: 600; color: #059669; margin: 0; font-size: 1.1rem;">Authentic Video</p>
                <p style="color: #475569; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Appears to be genuine</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Key metrics with enhanced cards
        st.markdown("#### 📈 Technical Metrics")
        
        fake_prob = result['fake_prob'] * 100
        temp_inconsistency = result['temporal_inconsistency']
        frame_entropy = result['frame_entropy']
        
        metrics_html = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div class="metric-card">
                <p style="color: #475569; margin: 0; font-size: 1rem; font-weight: 500;">Manipulation Probability</p>
                <p style="color: #1e293b; font-size: 1.8rem; font-weight: 600; margin: 0.5rem 0;">{fake_prob:.1f}%</p>
            </div>
            <div class="metric-card">
                <p style="color: #475569; margin: 0; font-size: 1rem; font-weight: 500;">Movement Consistency</p>
                <p style="color: #1e293b; font-size: 1.8rem; font-weight: 600; margin: 0.5rem 0;">{temp_inconsistency:.3f}</p>
            </div>
            <div class="metric-card" style="grid-column: 1 / -1;">
                <p style="color: #475569; margin: 0; font-size: 1rem; font-weight: 500;">Frame Detail Level</p>
                <p style="color: #1e293b; font-size: 1.8rem; font-weight: 600; margin: 0.5rem 0;">{frame_entropy:.2f} bits</p>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

    # Analysis visualization section
    if os.path.exists("analysis_report.png"):
        st.markdown("""
        <div class="result-card">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">📈 Frame-by-Frame Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image("analysis_report.png", use_container_width=True)
        
    # Enhanced technical explanation
    with st.expander("🔬 Technical Analysis Details", expanded=False):
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; border: 1px solid #e2e8f0; color: #000000;">
        
        ### 📊 Understanding the Analysis
        
        #### **🎯 Manipulation Probability (Blue Line)**
        - **Range:** 0.0 (definitely real) to 1.0 (definitely altered)
        - **Interpretation:** 
            - Values **> 0.7**: Strong evidence of manipulation
            - Values **0.3-0.7**: Uncertain, requires manual review
            - Values **< 0.3**: Likely authentic content
        
        #### **⏱️ Movement Consistency**
        - **What it measures:** Frame-to-frame movement stability
        - **Threshold:** Values **> 0.15** indicate potential editing
        - **Why it matters:** Natural videos have smooth transitions
        
        #### **🌈 Frame Detail Level (Green Line)**
        - **Range:** Typically 4.0-8.0 bits per pixel
        - **Natural videos:** Usually **> 6.5 bits** (rich detail)
        - **Altered content:** Often **< 6.0 bits** (over-smoothed)
        
        ### 🚨 Warning Signs
        - High manipulation probability with low movement consistency
        - Sudden detail drops in specific frames  
        - Unusual prediction patterns suggesting edits
        
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced warnings section
    warnings = []
    if result['temporal_inconsistency'] > 0.15:
        warnings.append("High movement inconsistency detected - Possible frame editing")
    if result['frame_entropy'] < 6.5 and result['prediction'] == "Fake":
        warnings.append("Low detail level detected - Unnatural textures")
    
    if warnings:
        st.markdown("""
        <div style="background: #fef3c7; border: 1px solid #f59e0b33; border-radius: 12px; padding: 1rem; margin: 1rem 0;">
            <h4 style="color: #f59e0b; margin-bottom: 1rem; font-size: 1.1rem;">⚠️ Analysis Warnings</h4>
        </div>
        """, unsafe_allow_html=True)
        for warning in warnings:
            st.warning(f"⚠️ {warning}")

# Main app logic with enhanced flow
if uploaded_file is not None:
    # File info display
    file_size = len(uploaded_file.read()) / (1024 * 1024)  # MB
    uploaded_file.seek(0)  # Reset file pointer
    
    st.markdown(f"""
    <div class="result-card">
        <h4 style="color: #1e293b; margin-bottom: 1rem;">📁 File Information</h4>
        <p style="color: #475569;"><strong>Filename:</strong> {uploaded_file.name}</p>
        <p style="color: #475569;"><strong>Size:</strong> {file_size:.2f} MB</p>
        <p style="color: #475569;"><strong>Type:</strong> {uploaded_file.type}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Save uploaded file
    video_path = save_uploaded_file(uploaded_file)
    
    if video_path:
        # Display video preview
        display_video(video_path)
        
        # Enhanced analyze button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Analysis", key="analyze", use_container_width=True):
                # Perform analysis
                result = analyze_video(video_path)
                
                if result:
                    # Show results
                    show_results(result)
                    
                    # Clean up
                    os.unlink(video_path)
                    if os.path.exists("analysis_report.png"):
                        os.unlink("analysis_report.png")
                else:
                    st.markdown("""
                    <div style="background: #fee2e2; border: 1px solid #ef444433; border-radius: 12px; padding: 1rem; margin: 1rem 0;">
                        <h4 style="color: #dc2626; margin-bottom: 0.5rem;">❌ Analysis Failed</h4>
                        <p style="color: #475569; margin: 0;">Unable to process this video. Please try another file or check the format.</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.error("Invalid video file. Please upload a supported format.")
else:
    # Enhanced demo section
    st.markdown("""
    <div class="result-card">
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎬</div>
            <h3 style="color: #1e293b; margin-bottom: 1rem;">Ready to Analyze Your Video</h3>
            <p style="color: #475569; font-size: 1.1rem; margin-bottom: 2rem;">
                Upload a video file to get started with our video authenticity detection
            </p>
            
            
      
    </div>
    """, unsafe_allow_html=True)