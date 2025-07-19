import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------
# Reference model: ViT from TensorFlow Hub
# https://tfhub.dev/sayakpaul/vit_b16_fe/1
# --------------------------------------

# Streamlit page config
st.set_page_config(page_title="Image Similarity Search", layout="wide")
st.title("🔍 Image Similarity Search using Vision Transformer (ViT)")

# Dataset image folder
IMAGE_FOLDER = "images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Load ViT model from TensorFlow Hub
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/sayakpaul/vit_b16_fe/1")

model = load_model()

# Preprocess uploaded image
def preprocess_image(image):
    """
    Preprocess image for ViT model:
    - Resize to 224x224
    - Normalize to [0,1]
    - Convert to float32
    - Add batch dimension
    """
    # Resize image
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Ensure 3 channels (RGB)
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[-1] == 4:  # RGBA
        image_array = image_array[:, :, :3]
    
    # Convert to tensor and add batch dimension
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    return image_tensor


# Load all dataset images and compute embeddings
@st.cache_data
def load_dataset_embeddings(image_folder=IMAGE_FOLDER):
    """Load dataset images and compute their embeddings"""
    embeddings = []
    paths = []

    if not os.path.exists(image_folder):
        st.warning(f"Dataset folder '{image_folder}' not found. Please add images to the dataset.")
        return np.array([]), []

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
    
    if not image_files:
        st.info(f"No images found in '{image_folder}' folder. Please add some images to the dataset.")
        return np.array([]), []

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(image_files):
        status_text.text(f"Processing {filename}... ({i+1}/{len(image_files)})")
        progress_bar.progress((i + 1) / len(image_files))
        
        path = os.path.join(image_folder, filename)
        try:
            image = Image.open(path).convert("RGB")
            img_tensor = preprocess_image(image)
            features = model(img_tensor).numpy()[0]
            embeddings.append(features)
            paths.append(path)
        except Exception as e:
            st.warning(f"Skipping {filename}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    if embeddings:
        st.success(f"✅ Loaded {len(embeddings)} images from dataset")
    
    return np.array(embeddings), paths

# Load dataset embeddings
dataset_embeddings, dataset_paths = load_dataset_embeddings()

# Perform similarity search
def get_similar_images(uploaded_image, top_k=5, similarity_threshold=0.3):
    """Find similar images using cosine similarity"""
    try:
        # Preprocess query image
        query_tensor = preprocess_image(uploaded_image)
        
        # Get embedding from model
        query_embedding = model(query_tensor).numpy()
        
        # Ensure we have dataset embeddings
        if len(dataset_embeddings) == 0:
            st.error("No images in dataset to compare against. Please add images to the dataset first.")
            return []
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, dataset_embeddings)[0]
        
        # Debug: Show similarity statistics
        st.write(f"🔍 **Similarity Range**: {similarities.min():.3f} to {similarities.max():.3f}")
        st.write(f"📊 **Average Similarity**: {similarities.mean():.3f}")
        
        # Filter by similarity threshold
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            st.warning(f"No images found with similarity >= {similarity_threshold:.2f}. Showing top results anyway.")
            valid_indices = np.arange(len(similarities))
        
        # Sort by similarity (highest first)
        valid_similarities = similarities[valid_indices]
        sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1]]
        
        # Get top k most similar images
        top_indices = sorted_indices[:min(top_k, len(sorted_indices))]
        
        results = [(dataset_paths[i], similarities[i]) for i in top_indices]
        
        # Debug: Show actual similarity scores
        st.write("**Top Similarity Scores:**")
        for i, (path, score) in enumerate(results[:3]):  # Show top 3
            filename = os.path.basename(path)
            st.write(f"{i+1}. {filename}: {score:.4f}")
        
        return results
    
    except Exception as e:
        st.error(f"Error during similarity search: {str(e)}")
        return []

# --- Sidebar: Dataset Management ---
st.sidebar.header("🗂️ Dataset Management")

# Display dataset info
if len(dataset_paths) > 0:
    st.sidebar.metric("Images in Dataset", len(dataset_paths))
else:
    st.sidebar.warning("Dataset is empty")

# Add image to dataset
st.sidebar.subheader("Add New Image")
dataset_file = st.sidebar.file_uploader(
    "Upload new image to dataset", 
    type=["jpg", "jpeg", "png", "bmp", "tiff"], 
    key="dataset"
)

if dataset_file:
    try:
        image = Image.open(dataset_file).convert("RGB")
        save_path = os.path.join(IMAGE_FOLDER, dataset_file.name)
        
        # Check if file already exists
        if os.path.exists(save_path):
            st.sidebar.warning(f"⚠️ File '{dataset_file.name}' already exists in dataset")
        else:
            image.save(save_path)
            st.sidebar.success(f"✅ Saved to dataset: {dataset_file.name}")
            st.sidebar.info("💡 Click 'Clear cache and rerun' below to update the dataset")
            
    except Exception as e:
        st.sidebar.error(f"❌ Failed to save image: {str(e)}")

# Refresh dataset button
if st.sidebar.button("🔄 Refresh Dataset"):
    st.cache_data.clear()
    st.rerun()

# --- Main: Query section ---
st.header("🔍 Search by Image")

# Upload query image
uploaded_file = st.file_uploader(
    "Upload an image to find similar images from the dataset", 
    type=["jpg", "jpeg", "png", "bmp", "tiff"]
)

if uploaded_file:
    try:
        # Load and display query image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Query Image", use_column_width=True)
        
        with col2:
            # Settings
            st.subheader("Search Settings")
            top_k = st.slider("Number of similar images to show", 1, min(10, len(dataset_paths)), 5)
            similarity_threshold = st.slider("Minimum similarity threshold", 0.0, 1.0, 0.3, 0.05)
            
            if len(dataset_paths) == 0:
                st.error("❌ No images in dataset. Please add some images first.")
            else:
                # Perform search
                if st.button("🔍 Find Similar Images", type="primary"):
                    with st.spinner("Searching for similar images..."):
                        results = get_similar_images(image, top_k, similarity_threshold)
                    
                    if results:
                        st.success(f"Found {len(results)} similar images!")
                        
                        # Display results
                        st.subheader("🎯 Most Similar Images")
                        
                        # Create columns based on number of results
                        cols = st.columns(min(len(results), 5))
                        
                        for i, (path, score) in enumerate(results):
                            col_idx = i % len(cols)
                            with cols[col_idx]:
                                try:
                                    result_image = Image.open(path)
                                    filename = os.path.basename(path)
                                    
                                    # Color-code based on similarity
                                    if score >= 0.8:
                                        border_color = "🟢"  # High similarity
                                    elif score >= 0.6:
                                        border_color = "🟡"  # Medium similarity
                                    else:
                                        border_color = "🔴"  # Low similarity
                                    
                                    st.image(
                                        result_image, 
                                        caption=f"{border_color} {filename}\nSimilarity: {score:.3f}", 
                                        use_column_width=True
                                    )
                                except Exception as e:
                                    st.error(f"Error loading {path}: {str(e)}")
                    else:
                        st.warning("No similar images found.")
        
    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")

# --- Information section ---
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info("""
This app uses a Vision Transformer (ViT) model to find similar images based on visual features.

**How it works:**
1. Upload images to build your dataset
2. Upload a query image
3. The model compares visual features using cosine similarity
4. View the most similar images from your dataset

**Supported formats:** JPG, JPEG, PNG, BMP, TIFF
""")

# Debug info (optional)
with st.expander("🔧 Debug Information"):
    st.write(f"Dataset folder: {IMAGE_FOLDER}")
    st.write(f"Images in dataset: {len(dataset_paths)}")
    if len(dataset_paths) > 0:
        st.write("Dataset files:")
        for path in dataset_paths[:10]:  # Show first 10
            st.write(f"- {os.path.basename(path)}")
        if len(dataset_paths) > 10:
            st.write(f"... and {len(dataset_paths) - 10} more")
    st.write(f"TensorFlow version: {tf.__version__}")
    st.write(f"Model loaded: {model is not None}")