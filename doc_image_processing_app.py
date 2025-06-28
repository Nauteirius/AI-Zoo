import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests

# Page configuration
st.set_page_config(page_title="Document Analysis", page_icon="📄")
st.title("Document Image Analysis")
st.sidebar.header("Options")

# Image processing functions
def load_image(image_file):
    """Load image from file uploader and convert to numpy array"""
    img = Image.open(image_file)
    return np.array(img)

def plot_histogram(img):
    """
    Plot histogram for grayscale or color image
    Returns matplotlib figure object
    """
    if len(img.shape) == 2:  # Grayscale image
        hist = cv2.calcHist([img], [0], None, [256], [0,256])
        plt.figure(figsize=(8,4))
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Pixel Count')
        plt.plot(hist, color='black')
        plt.xlim([0,256])
    
    else:  # Color image
        colors = ('b', 'g', 'r')
        plt.figure(figsize=(8,4))
        plt.title('Color Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Pixel Count')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0,256])
            plt.plot(hist, color=col)
            plt.xlim([0,256])
    
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def sharpen_image(img, kernel_size=5, sigma=1.0, amount=1.0, threshold=0):
    """
    Sharpen image using unsharp masking technique
    Parameters:
    - kernel_size: Size of Gaussian blur kernel
    - sigma: Standard deviation for Gaussian blur
    - amount: Sharpening strength
    - threshold: Only sharpen areas with contrast above threshold
    """
    # Convert to float for calculations
    img = img.astype(np.float32)
    
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    
    # Clip values to valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255)
    sharpened = sharpened.astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        sharpened = np.where(low_contrast_mask, img, sharpened)
    
    return sharpened

# User interface - File uploader
uploaded_file = st.file_uploader(
    "Upload document (JPG/PNG)", 
    type=["jpg", "jpeg", "png"]
)

# Default document image
default_image = "https://raw.githubusercontent.com/kevinam99/capturing-images-from-webcam-using-javascript-and-python-opencv/master/assets/invoice-sample.jpg"

if uploaded_file is not None:
    image = load_image(uploaded_file)
    # Remove alpha channel if exists
    if image.shape[-1] == 4:
        image = image[..., :3]
else:
    st.info("Using default invoice image")
    response = requests.get(default_image, stream=True)
    img = Image.open(response.raw)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    image = np.array(img)

# Functionality selection
option = st.radio(
    "Select functionality:",
    ("Histogram Analysis", "Image Sharpening"),
    horizontal=True
)

if option == "Histogram Analysis":
    st.header("Image Histogram Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Histogram")
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        fig = plot_histogram(grayscale)
        st.pyplot(fig)
    
    st.markdown("### Histogram Interpretation")
    st.write("""
    - **Dominant pixels**: Peaks indicate most frequent pixel values
    - **Document contrast**: Wide distribution = high contrast, narrow = low contrast
    - **Exposure issues**: Left-skewed = underexposed, right-skewed = overexposed
    """)

else:
    st.header("Image Sharpening")
    
    # Sharpening parameter controls
    with st.expander("Sharpening Parameters", expanded=True):
        col_params = st.columns(2)
        with col_params[0]:
            kernel_size = st.slider("Kernel Size", 3, 15, 5, 2)
            sigma = st.slider("Sigma (blur amount)", 0.1, 5.0, 1.0, 0.1)
        with col_params[1]:
            amount = st.slider("Sharpening Strength", 0.1, 3.0, 1.0, 0.1)
            threshold = st.slider("Sharpening Threshold", 0, 100, 0, 1)
    
    # Convert and sharpen
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = image
    
    sharpened = sharpen_image(grayscale, kernel_size, sigma, amount, threshold)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Sharpened Image")
        st.image(sharpened, use_container_width=True, clamp=True)
    
    # Difference comparison - FIXED DIMENSION MISMATCH
    st.subheader("Difference Visualization")
    
    # Create consistent 3-channel images for display
    if len(image.shape) == 2:
        original_display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        original_display = image[..., :3]  # Remove alpha channel
    else:
        original_display = image.copy()
    
    sharpened_display = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    
    # Compute and normalize difference
    difference = cv2.absdiff(grayscale, sharpened)
    difference = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
    difference = difference.astype(np.uint8)
    
    # Create color-mapped difference
    colored_diff = cv2.applyColorMap(difference, cv2.COLORMAP_JET)
    difference_rgb = cv2.cvtColor(difference, cv2.COLOR_GRAY2RGB)
    
    # Create comparison using Streamlit columns instead of numpy stacking
    st.info("Original vs Sharpened vs Difference")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original_display, caption="Original", use_container_width=True)
    with col2:
        st.image(sharpened_display, caption="Sharpened", use_container_width=True)
    with col3:
        st.image(difference_rgb, caption="Grayscale Difference", use_container_width=True)
    
    st.info("Colorized Difference Map")
    st.image(colored_diff, caption="Colorized Difference", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("""
**Usage instructions:**
1. Upload your document or use default
2. Select analysis functionality
3. For sharpening, adjust parameters
""")

# Add visual indicator for active tab
st.markdown(
    f"<style>div[role='radiogroup'] > label[data-value='{option}'] {{ background-color: #f0f2f6; }}</style>",
    unsafe_allow_html=True
)