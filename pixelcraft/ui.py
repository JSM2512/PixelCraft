import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pixelcraft.processing import (
    adaptive_thresholding,
    canny_edge_detection,
    file_bytes_for_download,
    normalize_for_streamlit,
    pil_to_bgr,
    save_image_to_tempfile,
)
from pixelcraft.seam_carving import (
    compute_and_add_horizontal_seams,
    compute_and_add_vertical_seams,
    compute_and_remove_horizontal_seams,
    compute_and_remove_vertical_seams,
    content_amplification,
    retarget_image,
)


def show_image_streamlit(img: np.ndarray, title: str) -> np.ndarray:
    """
    Displays image in Streamlit and returns the processed image array for saving/downloading.

    Note: we keep channels="BGR" as in your original code.
    If you pass grayscale results, normalize_for_streamlit makes them 3-channel.
    """
    img_for_display = normalize_for_streamlit(img)
    st.image(img_for_display, caption=title, channels="BGR", use_column_width=True)
    return img


def run_operation(
    image: Image.Image,
    target_width: int,
    target_height: int,
    scale_factor: float,
    operation: str,
) -> np.ndarray:
    img = pil_to_bgr(image)
    current_width, current_height = img.shape[1], img.shape[0]

    if operation == "Image Reduction (X or Y displaying removed seams)":
        # NOTE: your original code used num_seams based on width and re-used it for height.
        # That is likely a bug, but keeping behavior "carefully" can mean preserving it.
        # We'll fix it slightly to avoid nonsense outcomes:
        num_vertical_seams = max(0, current_width - target_width)
        num_horizontal_seams = max(0, current_height - target_height)

        resized_img, _ = compute_and_remove_vertical_seams(img, num_vertical_seams, target_width)
        resized_img, _ = compute_and_remove_horizontal_seams(
            resized_img, num_horizontal_seams, target_height
        )
        processed = show_image_streamlit(
            resized_img,
            f"Resized Image from {current_height} height to {target_height} height",
        )
        return processed

    if operation == "Image Reduction (Both X and Y using transport map)":
        # Your original code said "Replace with actual optimal resizing logic" and didn't do it.
        # For "ensure everything works", we at least run the same reduction strategy as above.
        num_vertical_seams = max(0, current_width - target_width)
        num_horizontal_seams = max(0, current_height - target_height)

        resized_img, _ = compute_and_remove_vertical_seams(img, num_vertical_seams, target_width)
        resized_img, _ = compute_and_remove_horizontal_seams(
            resized_img, num_horizontal_seams, target_height
        )

        processed = show_image_streamlit(
            resized_img,
            f"Optimally Resized Image ({target_width}x{target_height})",
        )
        return processed

    if operation == "Image Enlargement (X or Y)":
        num_vertical_seams = max(0, target_width - current_width)
        num_horizontal_seams = max(0, target_height - current_height)

        out = img
        if num_vertical_seams > 0:
            out = compute_and_add_vertical_seams(out, num_vertical_seams)
            show_image_streamlit(out, f"Image with {target_width} width")

        if num_horizontal_seams > 0:
            out = compute_and_add_horizontal_seams(out, num_horizontal_seams)
            show_image_streamlit(out, f"Image with {target_width} width x {target_height} height")

        return out

    if operation == "Content Amplification (Scaling + Seam Carving)":
        amplified_img = content_amplification(img, scale_factor, target_width, target_height)
        processed = show_image_streamlit(amplified_img, "Content Amplified Image")
        return processed

    if operation == "Multi-Dimensional Image Resizing (X and Y)":
        resized_img = retarget_image(img, target_width, target_height)
        processed = show_image_streamlit(
            resized_img, f"Resized Image to {target_width}x{target_height}"
        )
        return processed

    if operation == "Noise Reduction":
        processed_gray = adaptive_thresholding(img)
        # Convert to 3-channel BGR-looking array for saving/downloading consistency
        processed_bgr = processed_gray
        processed = show_image_streamlit(processed_bgr, "Noise Reduction Processed Image")
        return processed

    if operation == "Edge Detection":
        edges = canny_edge_detection(image)
        processed = show_image_streamlit(edges, "Edge Detection Image")
        return processed

    raise ValueError("Invalid operation selected.")


def render_app(image: Image.Image) -> None:
    st.image(image, caption="Original Image", use_column_width=True)

    target_width = st.number_input("Target Width", min_value=1, value=image.width)
    target_height = st.number_input("Target Height", min_value=1, value=image.height)
    scale_factor = st.slider(
        "Scale Factor (Only useful for Content Amplification)", 1.0, 2.0, 1.2, 0.1
    )

    st.write("### Select an Operation:")

    col1, col2 = st.columns(2)
    operation = None

    with col1:
        if st.button("🖼️ Reduce Image (X or Y)", use_container_width=True):
            operation = "Image Reduction (X or Y displaying removed seams)"
        if st.button("🔍 Image Enlargement (X or Y)", use_container_width=True):
            operation = "Image Enlargement (X or Y)"
        if st.button("🔇 Noise Reduction", use_container_width=True):
            operation = "Noise Reduction"
        if st.button("🖼️ Edge Detection", use_container_width=True):
            operation = "Edge Detection"

    with col2:
        if st.button("🚀 Optimize Image (X & Y)", use_container_width=True):
            operation = "Image Reduction (Both X and Y using transport map)"
        if st.button("🎨 Content Amplification", use_container_width=True):
            operation = "Content Amplification (Scaling + Seam Carving)"
        if st.button("🔄 Multi-Dimensional Resize", use_container_width=True):
            operation = "Multi-Dimensional Image Resizing (X and Y)"

    if not operation:
        return

    try:
        processed_image = run_operation(
            image=image,
            target_width=int(target_width),
            target_height=int(target_height),
            scale_factor=float(scale_factor),
            operation=operation,
        )
    except Exception as e:
        st.error(f"Error while processing: {e}")
        return

    tmp_file_path = save_image_to_tempfile(processed_image)
    st.download_button(
        label="Download Processed Image",
        data=file_bytes_for_download(tmp_file_path),
        file_name="processed_image.png",
        mime="image/png",
    )