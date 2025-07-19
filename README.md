
---


# 🧠 Image Similarity Search using Vision Transformer (ViT)

This project is a Streamlit web app that performs **image similarity search** using **Vision Transformer (ViT)** embeddings. Upload an image, and the model finds the most visually similar images from your dataset based on their ViT-generated embeddings.

## 🚀 Features

- Upload an image to search
- Compare it with a dataset of stored images
- Uses **ViT (Vision Transformer)** from TensorFlow Hub
- Computes **cosine similarity** between embeddings
- Returns top N similar images with similarity scores
- Easy-to-use **Streamlit UI**

## 🖼️ Demo

<img src="https://github.com/nabeelalikhan0/Image-Similarity/blob/main/demo.png" width="100%"/>

---

## 🧰 Tech Stack

- Python 🐍
- TensorFlow + Keras
- TensorFlow Hub
- NumPy
- Scikit-learn
- PIL (Pillow)
- Streamlit

---

## 📁 Project Structure

```bash
.
├── app.py                    # Streamlit app
├── extract_embeddings.py     # Script to precompute image embeddings
├── image_embeddings.npy      # Saved ViT embeddings (generated)
├── images/                   # Folder containing dataset images
├── requirements.txt          # Dependencies
└── README.md                 # This file


---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-similarity-vits.git
cd image-similarity-vits
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Images

Place all your dataset images inside the `images/` folder (create if not exists).

### 4. Extract Embeddings (Run Once)

```bash
python extract_embeddings.py
```

This script uses ViT to generate and save embeddings in `image_embeddings.npy`.

### 5. Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Loads ViT model from TensorFlow Hub
2. Precomputes embeddings for your dataset images
3. Accepts a query image through the UI
4. Extracts its embedding
5. Computes **cosine similarity** between the query image and dataset embeddings
6. Displays top N most similar images

---

## ⚠️ Notes

* Ensure your images are clear and relevant to your query domain.
* Preprocessing uses 224x224 image resizing to match ViT input.

---

## 📦 Requirements

```
tensorflow
tensorflow-hub
scikit-learn
numpy
pillow
streamlit
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 📸 Example Result

> Query Image vs Top-5 Most Similar Images from Dataset

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Nabeel Ali Khan**
Backend & AI Developer | Ethical Hacker
GitHub: [@nabeel03103n](https://github.com/nabeel03103n)

---

