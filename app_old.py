import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np
from skimage import util
from skimage import io, color
from skimage.feature import canny
import io
import tempfile

# Calculate the energy map of the image using gradient magnitude
def compute_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.absolute(gradient_x) + np.absolute(gradient_y)
    
    return energy

# Locating the horizontal seam with the least energy using dynamic programming.
def find_horizontal_seam(energy):   
    rows, cols = energy.shape
    seam = np.zeros(cols, dtype=np.int32)
    seam_energy = energy.copy()
    # Dynamic programming
    for col in range(1, cols):
        for row in range(rows):
            min_energy = seam_energy[row, col - 1]  ###  immediate left
            if row > 0:
                min_energy = min(min_energy, seam_energy[row - 1, col - 1])  ### top left
            if row < rows - 1:
                min_energy = min(min_energy, seam_energy[row + 1, col - 1])  ### bottom left
            seam_energy[row, col] += min_energy

    # Here we perform backtracking to find the seam path
    seam[-1] = np.argmin(seam_energy[:, -1])
    for col in range(cols - 2, -1, -1):
        prev_row = seam[col + 1]
        min_row = prev_row
        if prev_row > 0 and seam_energy[prev_row - 1, col] < seam_energy[min_row, col]:
            min_row = prev_row - 1
        if prev_row < rows - 1 and seam_energy[prev_row + 1, col] < seam_energy[min_row, col]:
            min_row = prev_row + 1
        seam[col] = min_row
    
    return seam

# Removing a horizontal seam from the image. This reduces the width of image by 1.
def remove_horizontal_seam(img, seam):
    rows, cols, _ = img.shape
    img_removed = np.zeros((rows - 1, cols, 3), dtype=np.uint8)  # New image with one less row
    for col in range(cols):
        row = seam[col]
        # Copying all rows except the one in the seam
        img_removed[:row, col] = img[:row, col]
        img_removed[row:, col] = img[row + 1:, col]
    
    return img_removed

# This method will draw all the horizontal seams in red
def highlight_horizontal_seams(img, seams):
    img_with_seams = img.copy()
    for seam in seams:
        for col in range(len(seam)):
            row = seam[col]
            img_with_seams[row, col] = [0, 0, 255]  # Color red
    return img_with_seams

# Locating the vertical seam with the least energy using dynamic programming.
def find_vertical_seam(energy):  
    rows, cols = energy.shape
    seam = np.zeros(rows, dtype=np.int32)
    seam_energy = energy.copy()
    # Dynamic programming
    for row in range(1, rows):
        for col in range(cols):
            min_energy = seam_energy[row - 1, col]
            if col > 0:
                min_energy = min(min_energy, seam_energy[row - 1, col - 1])
            if col < cols - 1:
                min_energy = min(min_energy, seam_energy[row - 1, col + 1])
            seam_energy[row, col] += min_energy

    # Here we perform backtracking to find the seam path
    seam[-1] = np.argmin(seam_energy[-1])
    for row in range(rows - 2, -1, -1):
        prev_col = seam[row + 1]
        min_col = prev_col
        if prev_col > 0 and seam_energy[row, prev_col - 1] < seam_energy[row, min_col]:
            min_col = prev_col - 1
        if prev_col < cols - 1 and seam_energy[row, prev_col + 1] < seam_energy[row, min_col]:
            min_col = prev_col + 1
        seam[row] = min_col
    
    return seam

# Removing a vertical seam from the image. This reduces the width of image by 1.
def remove_vertical_seam(img, seam):
    rows, cols, _ = img.shape
    img_removed = np.zeros((rows, cols - 1, 3), dtype=np.uint8)  # New image with one less column
    for row in range(rows):
        col = seam[row]
        # Copying all columns except the one in the seam
        img_removed[row, :col] = img[row, :col]  
        img_removed[row, col:] = img[row, col + 1:] 
    return img_removed

# This method will draw all the vertical seams in red
def highlight_vertical_seams(img, seams):
    img_with_seams = img.copy()
    for seam in seams:
        for row in range(len(seam)):
            col = seam[row]
            img_with_seams[row, col] = [0, 0, 255]  # Color red
    return img_with_seams

# Display the image
def display_image(img, title="Image"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Display the energy map
def display_energy_map(energy):
    plt.imshow(energy)
    plt.title('Energy Map')
    plt.axis('off')
    plt.show()

# Remove vertical seams from the image to reduce its width.
def compute_and_remove_vertical_seams(img, num_seams, target_width):
    seams = []
    # Compute and store all seams
    for _ in range(num_seams):
        energy = compute_energy(img)
        # Finding least energy vertical seam and remove it.
        seam = find_vertical_seam(energy)
        seams.append(seam)
        img = remove_vertical_seam(img, seam)
        # We will stop when we get our desired width
        if img.shape[1] <= target_width:
            break
    
    return img, seams

# Remove horizontal seams from the image to reduce its height.
def compute_and_remove_horizontal_seams(img, num_seams, target_height):
    seams = []
    # Compute and store all seams
    for _ in range(num_seams):
        # Compute the energy map
        energy = compute_energy(img)
        # Finding least energy horizontal seam and remove it.
        seam = find_horizontal_seam(energy)
        seams.append(seam)
        img = remove_horizontal_seam(img, seam)
        # We will stop when we get our desired height
        if img.shape[0] <= target_height:
            break
    
    return img, seams

# Calculating the transport map to optimize the order of seam removal.
def compute_transport_map(img, target_width, target_height):
    original_height, original_width = img.shape[:2]
    r, c = original_height - target_height, original_width - target_width
    
    # Initialize the transport map with high initial values
    T = np.full((r + 1, c + 1), np.inf)
    T[0, 0] = 0 
    # Dynamic progrmming
    for i in range(r + 1):
        for j in range(c + 1):
            if i > 0: 
                T[i, j] = min(T[i, j], T[i - 1, j] + np.sum(compute_energy(remove_horizontal_seam(img, find_horizontal_seam(compute_energy(img))))))
            if j > 0:  
                T[i, j] = min(T[i, j], T[i, j - 1] + np.sum(compute_energy(remove_vertical_seam(img, find_vertical_seam(compute_energy(img))))))

    return T

# Here we backtrack through the transport map to get the optimal seam removal order.
def backtrack_seam_order(T):
    r, c = T.shape
    r, c = r - 1, c - 1
    seam_order = []
    # Backtracking from the target dimensions to the original dimensions
    while r > 0 or c > 0:
        if r > 0 and (c == 0 or T[r - 1, c] < T[r, c - 1]):
            seam_order.append('horizontal')
            r -= 1
        else:
            seam_order.append('vertical')
            c -= 1
    return seam_order[::-1]  # Reversing the list to get the correct order

# In this method, we resize the image to target dimensions using an optimal seam removal strategy.
def resize_image_optimal(img, target_width, target_height):
    T = compute_transport_map(img, target_width, target_height)
    seam_order = backtrack_seam_order(T)

    for seam_type in seam_order:
        energy = compute_energy(img)
        if seam_type == 'horizontal':
            seam = find_horizontal_seam(energy)
            img = remove_horizontal_seam(img, seam)
        else:
            seam = find_vertical_seam(energy)
            img = remove_vertical_seam(img, seam)

    return img


# This method will add a vertical seam to the image by duplicating the pixels along the seam.
def add_vertical_seam(img, seam):
    rows, cols, _ = img.shape
    img_added = np.zeros((rows, cols + 1, 3), dtype=np.uint8)

    for row in range(rows):
        col = seam[row]
        for ch in range(3):  
            img_added[row, :col, ch] = img[row, :col, ch]  
            img_added[row, col, ch] = img[row, col, ch]  
            img_added[row, col + 1:, ch] = img[row, col:, ch]  

    return img_added

# This method will add a horizontal seam to the image by duplicating the pixels along the seam.
def add_horizontal_seam(img, seam):
    rows, cols, _ = img.shape
    img_added = np.zeros((rows + 1, cols, 3), dtype=np.uint8)

    for col in range(cols):
        row = seam[col]
        for ch in range(3):  
            img_added[:row, col, ch] = img[:row, col, ch]  
            img_added[row, col, ch] = img[row, col, ch]  
            img_added[row + 1:, col, ch] = img[row:, col, ch]  

    return img_added

# Calculate and add num_seams vertical seams to enlarge the image.
def compute_and_add_vertical_seams(img, num_seams):
    for _ in range(num_seams):
        energy = compute_energy(img)
        seam = find_vertical_seam(energy)
        img = add_vertical_seam(img, seam)
    return img

# Calculate and add num_seams horizontal seams to enlarge the image.
def compute_and_add_horizontal_seams(img, num_seams):
    for _ in range(num_seams):
        energy = compute_energy(img)
        seam = find_horizontal_seam(energy)
        img = add_horizontal_seam(img, seam)
    return img


# this method will amplify image content while preserving its size.
def content_amplification(img, scale_factor, target_width, target_height):
    # Scaling the image up
    scaled_width = int(img.shape[1]*scale_factor)
    scaled_height = int(img.shape[0]*scale_factor)
    scaled_img = cv2.resize(img, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    print(f"Scaled Image Dimensions: {scaled_img.shape[1]}x{scaled_img.shape[0]}")
    # Using seam carving to reduce the scaled image to the original size
    amplified_img = resize_image_optimal(scaled_img, target_width, target_height)
    
    return amplified_img

# Here we retarget the image to a new size by removing seams.
def retarget_image(img, new_width, new_height):
    resized_img = img.copy()
    # Reduce width
    while resized_img.shape[1] > new_width:
        energy = compute_energy(resized_img)
        vertical_seam = find_vertical_seam(energy)
        resized_img = remove_seam(resized_img, vertical_seam, axis=1)
    # Reduce height
    while resized_img.shape[0] > new_height:
        energy = compute_energy(resized_img)
        horizontal_seam = find_horizontal_seam(energy)
        resized_img = remove_seam(resized_img, horizontal_seam, axis=0)
    # Expand width
    while resized_img.shape[1] < new_width:
        energy = compute_energy(resized_img)
        vertical_seam = find_vertical_seam(energy)
        resized_img = add_seam(resized_img, vertical_seam, axis=1)
    # Expand height
    while resized_img.shape[0] < new_height:
        energy = compute_energy(resized_img)
        horizontal_seam = find_horizontal_seam(energy)
        resized_img = add_seam(resized_img, horizontal_seam, axis=0)

    return resized_img

def remove_seam(img, seam, axis):
    if axis == 1:  # Vertical seam
        return np.array([np.delete(row, seam[i], axis=0) for i, row in enumerate(img)])
    elif axis == 0:  # Horizontal seam
        return np.delete(img, seam, axis=0)

def add_seam(img, seam, axis):
    if axis == 1:  # Vertical seam
        new_img = []
        for i, row in enumerate(img):
            seam_idx = seam[i]
            new_pixel = np.mean(img[i, seam_idx-1:seam_idx+1], axis=0) if 0 < seam_idx < img.shape[1] - 1 else img[i, seam_idx]
            new_row = np.insert(row, seam_idx, new_pixel, axis=0)
            new_img.append(new_row)
        return np.array(new_img)
    
    elif axis == 0:  # Horizontal seam
        new_img = img.copy()
        for j in range(img.shape[1]):
            seam_idx = seam[j]
            new_pixel = np.mean(img[seam_idx-1:seam_idx+1, j], axis=0) if 0 < seam_idx < img.shape[0] - 1 else img[seam_idx, j]
            new_img = np.insert(new_img, seam_idx, new_pixel, axis=0)
        return new_img



def adaptive_thresholding(image):
    # Ensure the image is grayscale
    if len(image.shape) == 3:  # If the image is not grayscale (3D)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Parameters for the adaptive thresholding
    M, N = 5, 5
    g = image  # The grayscale image

    # Padding to handle the borders
    padded_image = np.pad(g, ((N//2, N//2), (M//2, M//2)), 'reflect')

    # Initialize zero-filled matrices for local variance and local mean
    lVar = np.zeros(g.shape)
    lM = np.zeros(g.shape)

    # Compute the local variance and mean with a 5x5 window
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            window = padded_image[i:i+M, j:j+N]
            lM[i, j] = np.mean(window)
            lVar[i, j] = np.var(window)

    # Estimate the noise variance as the mean of all local variances
    nVar = np.sum(lVar) / (g.shape[0] * g.shape[1])
    lVar = np.maximum(lVar, nVar)
    ratio = nVar / lVar

    # Adaptive filtering formula
    adaptive_filtered_image = g - ratio * (g - lM)

    return adaptive_filtered_image


def canny_edge_detection(image):
    # Convert PIL Image to numpy array
    image = np.array(image)
    
    # Check if the image is color (3 channels) or grayscale (1 channel)
    if len(image.shape) == 3:  # Color image (3 channels)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image  # It's already a grayscale image
    
    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred_image, 100, 200)  # Lower and upper thresholds can be adjusted
    
    return edges




def display_image(img, title="Image"):
    # Check if the image is grayscale (2D) or color (3D)
    if len(img.shape) == 2:  # Grayscale image (2D)
        img = np.stack([img] * 3, axis=-1)  # Convert grayscale to 3-channel image
    elif img.shape[2] == 1:  # Single channel image (e.g., RGBA with no color channels)
        img = np.concatenate([img] * 3, axis=-1)  # Convert to 3 channels by duplicating the channel
    
    # Normalize the image to [0.0, 1.0] if it's in the [0, 255] range
    if img.max() > 1:
        img = img / 255.0  # Rescale to [0.0, 1.0] if the max value is greater than 1

    # Display the image using Streamlit
    st.image(img, caption=title, channels="BGR", use_column_width=True)
    
    return img  # Return the processed image for download

def save_image_to_tempfile(image):
    # Ensure image is in uint8 format (if it's float, rescale it)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)  # Rescale to [0, 255] and convert to uint8

    # Convert NumPy array back to PIL image
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image_pil.save(tmp_file, format='PNG')
        tmp_file_path = tmp_file.name
    
    return tmp_file_path

def main(image, target_width, target_height, scale_factor, operation):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV format
    current_width, current_height = img.shape[1], img.shape[0]
    
    if operation == "Image Reduction (X or Y displaying removed seams)":
        # #### IMAGE REDUCTION EITHER X OR Y (RUN)
        num_seams = current_width - target_width
        resized_img, seams = compute_and_remove_vertical_seams(img, num_seams, target_width)
        resized_img, seams = compute_and_remove_horizontal_seams(img, num_seams, target_height)
        processed_image = display_image(resized_img, f"Resized Image from {current_height} height to {target_height} height")
    
    elif operation == "Image Reduction (Both X and Y using transport map)":
        resized_img = img  # Replace with actual optimal resizing logic
        processed_image = display_image(resized_img, f"Optimally Resized Image ({target_width}x{target_height})")
    
    elif operation == "Image Enlargement (X or Y)":
        num_vertical_seams = target_width - current_width
        num_horizontal_seams = target_height - current_height

        if num_vertical_seams > 0:
            img_with_vertical_seams = compute_and_add_vertical_seams(img, num_vertical_seams)
            processed_image = display_image(img_with_vertical_seams, f"Image with {target_width} width")

        if num_horizontal_seams > 0:
            img_with_horizontal_seams = compute_and_add_horizontal_seams(img_with_vertical_seams, num_horizontal_seams)
            processed_image = display_image(img_with_horizontal_seams, f"Image with {target_width} width x {target_height} height")
    
    elif operation == "Content Amplification (Scaling + Seam Carving)":
        amplified_img = content_amplification(img, scale_factor, target_width, target_height)
        processed_image = display_image(amplified_img, "Content Amplified Image")
    
    elif operation == "Multi-Dimensional Image Resizing (X and Y)":
        resized_img = retarget_image(img, target_width, target_height)
        processed_image = display_image(resized_img, f"Resized Image to {target_width}x{target_height}")
    
    elif operation == "Noise Reduction":
        processed_image = adaptive_thresholding(img)
        processed_image = display_image(processed_image, "Noise Reduction Processed Image")

    elif operation == "Edge Detection":
        processed_image = canny_edge_detection(image)
        processed_image = display_image(processed_image, "Edge Detection Image")

    else:
        st.error("Invalid operation selected.")
        return
    
    # Save the processed image to a temporary file
    tmp_file_path = save_image_to_tempfile(processed_image)  
    
    # Provide a download button for the user to download the processed image
    st.download_button(
        label="Download Processed Image",
        data=open(tmp_file_path, "rb").read(),
        file_name="processed_image.png",
        mime="image/png"
    )

st.markdown("# PixelCraft")
st.markdown("### Transform Your Images with Powerful Processing Tools")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "tif", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    target_width = st.number_input("Target Width", min_value=1, value=image.width)
    target_height = st.number_input("Target Height", min_value=1, value=image.height)
    scale_factor = st.slider("Scale Factor (Only useful for Content Amplification)", 1.0, 2.0, 1.2, 0.1)

    st.write("### Select an Operation:")

    col1, col2 = st.columns(2)

    operation = None  # Initialize operation variable

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

    if operation:
        main(image, target_width, target_height, scale_factor, operation)
