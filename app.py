"""
Complete Streamlit App for Civil Engineering Fault Detection - StructLens
Save this as app.py and run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import joblib
from datetime import datetime
from PIL import Image
import tempfile
import base64
from io import BytesIO
import subprocess
import requests
import warnings
warnings.filterwarnings('ignore')

# ============================================
# FUNCTIONS FROM YOUR JUPYTER NOTEBOOK
# ============================================

def get_llm_analysis(crack_analysis, ml_result, ml_confidence, metadata):
    """
    Get analysis from LLM (using Ollama)
    """
    
    # Prepare prompt for LLM
    prompt = f"""You are a senior civil engineer. Analyze this structural fault:

    📊 IMAGE ANALYSIS RESULTS:
    - ML Model Prediction: {ml_result} (Confidence: {ml_confidence:.1f}%)
    - Visual Crack Analysis:
      * Number of cracks detected: {crack_analysis['crack_count']}
      * Total crack area: {crack_analysis['total_crack_area']} pixels
      * Maximum crack length: {crack_analysis['max_crack_length']:.1f} pixels
      * Severity level: {crack_analysis['severity']}

    📝 STRUCTURE INFORMATION:
    {metadata}

    Based on civil engineering principles, provide a DETAILED REPORT with:

    1. FAULT IDENTIFICATION:
       - Type of structural fault
       - Affected structural element
       - Immediate risk assessment

    2. CAUSE ANALYSIS:
       - Likely causes
       - Contributing factors
       - Estimated timeline of occurrence

    3. PREVENTION MEASURES:
       - Design improvements
       - Maintenance recommendations
       - Monitoring techniques

    4. REMEDIATION:
       - Recommended repair methods
       - Urgency level
       - Estimated complexity

    Format as a professional engineering report."""
    
    try:
        # Try using Ollama (local)
        result = subprocess.run(
            ['ollama', 'run', 'phi3:mini', prompt],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    
    except:
        try:
            # Fallback to Hugging Face API (free)
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
            
            response = requests.post(API_URL, json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 500}
            })
            
            if response.status_code == 200:
                return response.json()[0]['generated_text']
            else:
                return get_fallback_response(crack_analysis)
        except:
            return get_fallback_response(crack_analysis)

def get_fallback_response(crack_analysis):
    """Fallback response if LLM is unavailable"""
    
    severity = crack_analysis['severity']
    
    if severity == "High":
        return """ENGINEERING REPORT (FALLBACK ANALYSIS)

1. FAULT IDENTIFICATION:
   - Type: Severe structural cracking detected
   - Element: Concrete structural member
   - Risk: IMMEDIATE - Structure requires urgent inspection

2. CAUSE ANALYSIS:
   - Possible overloading or material fatigue
   - May indicate reinforcement corrosion
   - Could be due to settlement or thermal stress

3. PREVENTION:
   - Regular structural inspections
   - Load monitoring systems
   - Proper drainage and waterproofing

4. REMEDIATION:
   - Immediate shoring required
   - Consult structural engineer urgently
   - Consider epoxy injection or replacement"""
    
    elif severity == "Medium":
        return """ENGINEERING REPORT (FALLBACK ANALYSIS)

1. FAULT IDENTIFICATION:
   - Type: Moderate cracking observed
   - Element: Concrete structure
   - Risk: MONITOR - Schedule detailed inspection

2. CAUSE ANALYSIS:
   - Likely shrinkage or minor settlement
   - Possible environmental factors
   - Normal wear and tear

3. PREVENTION:
   - Monitor crack propagation
   - Seal surface cracks
   - Improve drainage if external

4. REMEDIATION:
   - Fill cracks with appropriate sealant
   - Schedule follow-up in 3 months
   - Document for future reference"""
    
    else:
        return """ENGINEERING REPORT (FALLBACK ANALYSIS)

1. FAULT IDENTIFICATION:
   - Type: Minor surface imperfections
   - Element: Concrete surface
   - Risk: LOW - No immediate concern

2. CAUSE ANALYSIS:
   - Likely surface shrinkage
   - Normal curing-related cracks
   - Cosmetic in nature

3. PREVENTION:
   - Regular maintenance
   - Proper curing procedures
   - Quality control in construction

4. REMEDIATION:
   - Monitor during routine inspections
   - No immediate action required
   - Document in maintenance log"""

def extract_features(image_path):
    """
    Extract features from image for crack detection
    """
    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return None
        
        # Resize for consistency
        img = cv2.resize(img, (128, 128))
        
        features = []
        
        # 1. Basic statistical features
        features.append(np.mean(img))           # Mean intensity
        features.append(np.std(img))            # Standard deviation
        features.append(np.percentile(img, 75) - np.percentile(img, 25))  # IQR
        features.append(np.median(img))         # Median
        features.append(np.sum(img > 200) / (128*128))  # Bright pixel ratio
        features.append(np.sum(img < 50) / (128*128))   # Dark pixel ratio
        
        # 2. Edge features (important for crack detection)
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        features.append(edge_density)
        
        # Edge statistics
        features.append(np.mean(edges))
        features.append(np.std(edges))
        
        # 3. Texture features using Local Binary Pattern
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(img, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-6)  # Normalize
        features.extend(lbp_hist)
        
        # 4. HOG features
        from skimage.feature import hog
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), feature_vector=True)
        # Take statistics of HOG features
        features.append(np.mean(hog_features))
        features.append(np.std(hog_features))
        features.append(np.max(hog_features))
        features.append(np.min(hog_features))
        
        # 5. Fourier transform features (frequency domain)
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        features.append(np.mean(magnitude_spectrum))
        features.append(np.std(magnitude_spectrum))
        
        return np.array(features)
    
    except Exception as e:
        print(f"❌ Error extracting features from {image_path}: {e}")
        return None

# ============================================
# STREAMLIT APP CODE
# ============================================

# Page configuration
st.set_page_config(
    page_title="StructLens - Civil Engineering Fault Detector",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .report-box {
        background-color: #F3F4F6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .severity-high {
        color: #DC2626;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .severity-medium {
        color: #F59E0B;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .severity-low {
        color: #10B981;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
    }
    .image-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# Load models
@st.cache_resource
def load_models():
    """Load trained models with caching"""
    try:
        model = joblib.load('crack_detector_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.warning(f"⚠️ Model not found: {e}. Please train the model first.")
        return None, None

# Function to analyze crack
def analyze_crack_image(image):
    """Analyze crack in image"""
    # Convert to numpy array if PIL image
    if isinstance(image, Image.Image):
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold for crack detection
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze cracks
    crack_areas = []
    crack_lengths = []
    
    # Draw contours on a copy for visualization
    img_with_contours = img.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter noise
            crack_areas.append(area)
            perimeter = cv2.arcLength(contour, True)
            crack_lengths.append(perimeter)
            # Draw contour in green
            cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 2)
    
    # Determine severity
    if len(crack_areas) > 0:
        max_area = max(crack_areas)
        if max_area > 1000:
            severity = "High"
            severity_color = "severity-high"
        elif max_area > 200:
            severity = "Medium"
            severity_color = "severity-medium"
        else:
            severity = "Low"
            severity_color = "severity-low"
        
        crack_count = len(crack_areas)
        total_area = sum(crack_areas)
        max_length = max(crack_lengths) if crack_lengths else 0
        avg_area = np.mean(crack_areas) if crack_areas else 0
    else:
        severity = "None"
        severity_color = ""
        crack_count = 0
        total_area = 0
        max_length = 0
        avg_area = 0
    
    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_with_contours_rgb = cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)
    
    return {
        'crack_count': crack_count,
        'total_crack_area': total_area,
        'avg_crack_area': avg_area,
        'max_crack_length': max_length,
        'severity': severity,
        'severity_color': severity_color,
        'contours': contours,
        'thresh': thresh,
        'gray': gray,
        'img_with_contours': img_with_contours_rgb,
        'original_rgb': img_rgb
    }

# Function to get ML prediction
def get_ml_prediction(image_path, model, scaler):
    """Get ML model prediction"""
    try:
        features = extract_features(image_path)
        if features is not None and model is not None and scaler is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
            ml_pred = model.predict(features_scaled)[0]
            ml_prob = model.predict_proba(features_scaled)[0]
            
            result = "Crack Detected" if ml_pred == 1 else "No Crack"
            confidence = max(ml_prob) * 100
            return result, confidence, ml_prob
    except Exception as e:
        st.warning(f"ML prediction error: {e}")
        pass
    return "N/A", 0, [0, 0]

# Function to generate download link
def get_download_link(text, filename, link_text):
    """Generate a download link for text data"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Main app
def main():
    # Header with StructLens name
    st.markdown('<h1 class="main-header">🏗️ StructLens</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #4B5563; font-size: 1.2rem;">Civil Engineering Fault Detection System</p>', 
                unsafe_allow_html=True)
    
    # Load models
    model, scaler = load_models()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/civil-engineering.png", 
                 width=100)
        st.markdown("## 🛠️ Controls")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📤 Upload Image", "📸 Take Photo", "📋 Sample Images"]
        )
        
        st.markdown("---")
        
        # Metadata inputs
        st.markdown("## 📝 Structure Metadata")
        location = st.text_input("Location", "Unknown Structure")
        structure_type = st.selectbox(
            "Structure Type",
            ["Bridge", "Building", "Dam", "Pavement", "Tunnel", "Wall", "Other"]
        )
        element_type = st.selectbox(
            "Structural Element",
            ["Pillar/Column", "Beam", "Wall", "Deck/Slab", "Foundation", "Surface", "Other"]
        )
        inspection_date = st.date_input("Inspection Date", datetime.now())
        environmental_conditions = st.multiselect(
            "Environmental Conditions",
            ["Coastal", "Industrial", "Urban", "Rural", "High Humidity", 
             "Freeze-Thaw", "High Traffic", "Chemical Exposure"]
        )
        notes = st.text_area("Additional Notes", "")
        
        st.markdown("---")
        
        # LLM Settings
        st.markdown("## 🤖 LLM Settings")
        use_llm = st.checkbox("Enable LLM Analysis", value=True)
        
        st.markdown("---")
        
        # History
        st.markdown("## 📋 Analysis History")
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.session_state.current_result = None
            st.success("History cleared!")
        
        if st.session_state.analysis_history:
            for i, item in enumerate(st.session_state.analysis_history[-5:]):
                st.text(f"{i+1}. {item['timestamp']} - Severity: {item['severity']}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📸 Input Image</h2>', 
                   unsafe_allow_html=True)
        
        # Image input based on selection
        image_file = None
        if input_method == "📤 Upload Image":
            image_file = st.file_uploader(
                "Choose an image...", 
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
                help="Upload a clear image of the structural fault (supports JPG, PNG, BMP, TIFF, WEBP)"
            )
            
        elif input_method == "📸 Take Photo":
            image_file = st.camera_input("Take a photo")
            
        else:  # Sample images
            st.info("Select a sample image to test:")
            sample_images = {
                "Sample 1 - Bridge Crack": "samples/bridge_crack.jpg",
                "Sample 2 - Wall Crack": "samples/wall_crack.jpg",
                "Sample 3 - Pavement Crack": "samples/pavement_crack.jpg",
                "Sample 4 - No Crack": "samples/no_crack.jpg"
            }
            
            selected_sample = st.selectbox("Choose sample:", list(sample_images.keys()))
            
            # Check if sample exists
            sample_path = sample_images[selected_sample]
            if os.path.exists(sample_path):
                image_file = sample_path
            else:
                st.warning("Sample images not found. Please upload your own image.")
        
        # Display uploaded image
        if image_file is not None:
            # Read image
            if isinstance(image_file, str):  # Path
                image = Image.open(image_file)
            else:  # Uploaded file
                image = Image.open(image_file)
            
            # Resize for display
            display_image = image.copy()
            display_image.thumbnail((400, 400))
            st.image(display_image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown(f"**Image Info:**")
            st.markdown(f"- Format: {image.format if image.format else 'Unknown'}")
            st.markdown(f"- Size: {image.size[0]} x {image.size[1]} pixels")
            st.markdown(f"- Mode: {image.mode}")
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Analysis Results</h2>', 
                   unsafe_allow_html=True)
        
        # Analyze button
        if image_file is not None and st.button("🚀 Analyze Structure", type="primary"):
            with st.spinner("🔄 Analyzing image... This may take a moment."):
                
                # Read image if it's a path
                if isinstance(image_file, str):
                    image = Image.open(image_file)
                
                # Save image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(tmp_file, format='JPEG')
                    temp_path = tmp_file.name
                
                # Step 1: Analyze crack
                crack_analysis = analyze_crack_image(image)
                
                # Step 2: Get ML prediction
                ml_result, ml_confidence, ml_probs = get_ml_prediction(
                    temp_path, model, scaler
                )
                
                # Step 3: Prepare metadata
                metadata = f"""
Location: {location}
Structure Type: {structure_type}
Element: {element_type}
Date: {inspection_date}
Environmental Conditions: {', '.join(environmental_conditions) if environmental_conditions else 'Not specified'}
Notes: {notes}
                """
                
                # Step 4: Get LLM analysis
                if use_llm:
                    with st.spinner("🤖 Getting LLM analysis..."):
                        llm_report = get_llm_analysis(crack_analysis, ml_result, 
                                                     ml_confidence, metadata)
                else:
                    llm_report = "LLM analysis disabled by user."
                
                # Compile results
                result = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'crack_analysis': crack_analysis,
                    'ml_result': ml_result,
                    'ml_confidence': ml_confidence,
                    'ml_probabilities': ml_probs,
                    'llm_report': llm_report,
                    'metadata': metadata,
                    'image_path': temp_path,
                    'severity': crack_analysis['severity']
                }
                
                st.session_state.current_result = result
                st.session_state.analysis_history.append(result)
                st.rerun()
        
        # Display results if available
        if st.session_state.current_result:
            result = st.session_state.current_result
            
            # Metrics in columns
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔢 Crack Count</h3>
                    <h2>{result['crack_analysis']['crack_count']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[1]:
                severity = result['crack_analysis']['severity']
                severity_class = result['crack_analysis'].get('severity_color', '')
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚠️ Severity</h3>
                    <h2 class="{severity_class}">{severity}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📏 Crack Area</h3>
                    <h2>{result['crack_analysis']['total_crack_area']:.0f} px²</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 ML Confidence</h3>
                    <h2>{result['ml_confidence']:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # ML Prediction
            if result['ml_result'] != "N/A":
                st.info(f"🤖 ML Model Prediction: **{result['ml_result']}** "
                       f"(Confidence: {result['ml_confidence']:.1f}%)")
            
            # Detailed Report
            st.markdown("### 📄 Detailed Engineering Report")
            with st.container():
                st.markdown(f"""
                <div class="report-box">
                    {result['llm_report']}
                </div>
                """, unsafe_allow_html=True)
            
            # Download buttons
            st.markdown("### 📥 Download Options")
            col_dl1, col_dl2 = st.columns(2)
            
            # Full report
            full_report = f"""
CIVIL ENGINEERING FAULT ANALYSIS REPORT - StructLens
===================================================
Generated: {result['timestamp']}

METADATA:
{result['metadata']}

ANALYSIS RESULTS:
- Cracks Detected: {result['crack_analysis']['crack_count']}
- Severity Level: {result['crack_analysis']['severity']}
- Total Crack Area: {result['crack_analysis']['total_crack_area']} pixels²
- Average Crack Area: {result['crack_analysis']['avg_crack_area']:.2f} pixels²
- Maximum Crack Length: {result['crack_analysis']['max_crack_length']:.2f} pixels
- ML Prediction: {result['ml_result']} ({result['ml_confidence']:.1f}% confidence)

DETAILED ENGINEERING REPORT:
{result['llm_report']}
            """
            
            with col_dl1:
                filename = f"fault_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                st.markdown(get_download_link(full_report, filename, "📄 Download Report"), 
                           unsafe_allow_html=True)
            
            with col_dl2:
                # Summary CSV
                summary_df = pd.DataFrame([{
                    'Timestamp': result['timestamp'],
                    'Crack Count': result['crack_analysis']['crack_count'],
                    'Severity': result['crack_analysis']['severity'],
                    'Total Area': result['crack_analysis']['total_crack_area'],
                    'ML Confidence': result['ml_confidence'],
                    'ML Result': result['ml_result']
                }])
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Visualization section (full width)
    if st.session_state.current_result:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Visual Analysis</h2>', 
                   unsafe_allow_html=True)
        
        result = st.session_state.current_result
        crack_analysis = result['crack_analysis']
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Original image with contours (from our function)
        axes[0, 0].imshow(crack_analysis['original_rgb'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Image with contours
        axes[0, 1].imshow(crack_analysis['img_with_contours'])
        axes[0, 1].set_title(f'Detected Cracks: {crack_analysis["crack_count"]}')
        axes[0, 1].axis('off')
        
        # Grayscale
        axes[0, 2].imshow(crack_analysis['gray'], cmap='gray')
        axes[0, 2].set_title('Grayscale')
        axes[0, 2].axis('off')
        
        # Threshold mask
        axes[1, 0].imshow(crack_analysis['thresh'], cmap='gray')
        axes[1, 0].set_title('Crack Detection Mask')
        axes[1, 0].axis('off')
        
        # Edge detection
        edges = cv2.Canny(crack_analysis['gray'], 50, 150)
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title('Edge Detection')
        axes[1, 1].axis('off')
        
        # Severity gauge
        severity = crack_analysis['severity']
        colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green', 'None': 'blue'}
        color = colors.get(severity, 'gray')
        
        axes[1, 2].pie([1], colors=[color], labels=[f'Severity: {severity}'], 
                       autopct='', startangle=90)
        axes[1, 2].set_title('Severity Level')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Clean up temp file
        try:
            os.unlink(result['image_path'])
        except:
            pass

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>🏗️ StructLens - Civil Engineering Fault Detection System</p>
        <p>📧 Contact: adamarchit08@gmail.com</p>
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()