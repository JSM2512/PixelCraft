# PixelCraft

PixelCraft is a Streamlit-based image manipulation tool that provides multiple classic image-processing operations, including **seam carving** (content-aware resizing), **content amplification**, **noise reduction**, and **edge detection**.

The app is designed to be flexible and modular: the Streamlit UI is separated from the image-processing and seam-carving logic.

---

## Features

- **Image reduction (content-aware)** via seam carving  
- **Image enlargement** via seam insertion  
- **Multi-dimensional retargeting** (resize width + height)  
- **Content amplification** (scale up then seam carve back to target size)  
- **Noise reduction** (adaptive thresholding-style filtering)  
- **Edge detection** (Canny)

---

## How to Run (Conda)

### 1) Create a conda environment
```bash
conda create -n pixelcraft python=3.10 -y
conda activate pixelcraft
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Start the Streamlit app
```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).

---

## Usage

1. Upload an image (`.jpg`, `.png`, `.jpeg`, `.tif`, `.bmp`)
2. Set:
   - target width
   - target height
   - scale factor (used for **Content Amplification**)
3. Choose an operation from the buttons
4. Download the processed result using the download button

---

## Project Structure (New Modular Layout)

```text
.
├── app.py                 # Streamlit entrypoint (thin wrapper that launches the UI)
├── app_old.py             # Previous single-file version (kept for reference)
├── requirements.txt
├── pixelcraft/            # Core package (application logic)
│   ├── processing.py      # Noise reduction, edge detection, image conversion helpers
│   ├── seam_carving.py    # Seam carving + resizing + retargeting algorithms
│   └── ui.py              # Streamlit UI + operation routing
├── notebooks/             # Jupyter notebooks (experiments / exploration)
├── images/                # Example/sample images (optional)
└── Report.pdf             # Project report/documentation
```
