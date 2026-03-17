import streamlit as st
from PIL import Image

from pixelcraft.ui import render_app


def main():
    st.markdown("# PixelCraft")
    st.markdown("### Transform Your Images with Powerful Processing Tools")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg", "tif", "bmp"],
    )

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)
    render_app(image)


if __name__ == "__main__":
    main()