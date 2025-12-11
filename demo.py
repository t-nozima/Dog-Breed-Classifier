import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐕",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Container - Clean White/Blue Theme */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
        color: #102a43;
    }
    
    /* Transparent Header (Removes Top White Rectangle) */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Header Text */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #3182ce 0%, #63b3ed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(16, 42, 67, 0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #486581; /* Slate Blue */
        font-weight: 400;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* File Uploader - Fully Transparent */
    section[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.0); /* Transparent */
        border: 2px dashed #bcccdc;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
        box-shadow: none;
    }
    
    section[data-testid="stFileUploader"] > div {
        background-color: transparent !important;
    }
    
    section[data-testid="stFileUploader"]:hover {
        border-color: #3182ce;
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Button Container */
    .stButton {
        display: block; /* Default block behavior */
        margin-top: 20px;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #3182ce 0%, #4299e1 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(49, 130, 206, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
        text-align: center;
    }
    
    /* Hover */
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(49, 130, 206, 0.4);
        background: linear-gradient(90deg, #2c5282 0%, #2b6cb0 100%);
    }
    
    /* Active */
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Result Card */
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .result-title {
        color: #627d98;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    .result-breed {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2b6cb0 0%, #4299e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    /* Image Container - Centering */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 4px solid #ffffff;
        margin: 0 auto;
    }
    
    /* Centering Helper */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    'n02088466-bloodhound',
    'n02109525-Saint_Bernard',
    'n02093647-Bedlington_terrier',
    'n02085936-Maltese_dog',
    'n02106662-German_shepherd',
    'n02106166-Border_collie',
    'n02096051-Airedale',
    'n02097298-Scotch_terrier',
    'n02091134-whippet',
    'n02091032-Italian_greyhound',
    'n02108915-French_bulldog',
    'n02108089-boxer',
    'n02091467-Norwegian_elkhound',
    'n02110063-malamute',
    'n02099712-Labrador_retriever',
    'n02099601-golden_retriever',
    'n02094433-Yorkshire_terrier',
    'n02105505-komondor',
    'n02088364-beagle',
    'n02106550-Rottweiler'
]

def format_breed_name(raw_name):
    """Format raw class name to readable string."""
    parts = raw_name.split('-', 1)
    if len(parts) > 1:
        name = parts[1]
    else:
        name = raw_name
    
    name = name.replace('_', ' ')
    return name.capitalize()

@st.cache_resource
def load_learner():
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def main():
    st.markdown('<h1 class="main-header">Dog Breed Identifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a photo to discover the breed instantly.</p>', unsafe_allow_html=True)

    with st.container():
        uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            
            should_predict = False
            with btn_col2:
                if st.button("Identify Breed", use_container_width=True):
                    should_predict = True
            
            if should_predict:
                with st.spinner("Analyzing image features..."):
                    model = load_learner()
                    
                    if model:
                        if image.mode != "RGB":
                            image = image.convert("RGB")

                        img_resized = image.resize((224, 224))
                        img_array = np.array(img_resized)
                        
                        img_array = img_array[..., ::-1]
                        
                        mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
                        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
                        
                        img_array = img_array.astype('float32')
                        img_array = (img_array - mean) / (std + 1e-7)
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        predictions = model.predict(img_array)
                        predicted_index = np.argmax(predictions)
                        confidence = np.max(predictions)
                        
                        if predicted_index < len(CLASS_NAMES):
                            raw_name = CLASS_NAMES[predicted_index]
                            formatted_name = format_breed_name(raw_name)
                        else:
                            formatted_name = f"Unknown (Index {predicted_index})"

                        st.markdown(f"""
                            <div class="result-card">
                                <div class="result-title">We detected a</div>
                                <div class="result-breed">{formatted_name}</div>
                                <div style="margin-top: 10px; font-size: 0.8rem; color: #888;">
                                    Confidence: {confidence:.2%}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()