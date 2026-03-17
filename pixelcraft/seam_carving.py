import cv2
import numpy as np


# Calculate the energy map of the image using gradient magnitude
def compute_energy(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.absolute(gradient_x) + np.absolute(gradient_y)
    return energy


# Locating the horizontal seam with the least energy using dynamic programming.
def find_horizontal_seam(energy: np.ndarray) -> np.ndarray:
    rows, cols = energy.shape
    seam = np.zeros(cols, dtype=np.int32)
    seam_energy = energy.copy()

    for col in range(1, cols):
        for row in range(rows):
            min_energy = seam_energy[row, col - 1]  # immediate left
            if row > 0:
                min_energy = min(min_energy, seam_energy[row - 1, col - 1])  # top-left
            if row < rows - 1:
                min_energy = min(min_energy, seam_energy[row + 1, col - 1])  # bottom-left
            seam_energy[row, col] += min_energy

    seam[-1] = int(np.argmin(seam_energy[:, -1]))
    for col in range(cols - 2, -1, -1):
        prev_row = seam[col + 1]
        min_row = prev_row
        if prev_row > 0 and seam_energy[prev_row - 1, col] < seam_energy[min_row, col]:
            min_row = prev_row - 1
        if prev_row < rows - 1 and seam_energy[prev_row + 1, col] < seam_energy[min_row, col]:
            min_row = prev_row + 1
        seam[col] = min_row

    return seam


# Removing a horizontal seam from the image. (Reduces height by 1.)
def remove_horizontal_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    rows, cols, _ = img.shape
    img_removed = np.zeros((rows - 1, cols, 3), dtype=np.uint8)
    for col in range(cols):
        row = seam[col]
        img_removed[:row, col] = img[:row, col]
        img_removed[row:, col] = img[row + 1 :, col]
    return img_removed


def highlight_horizontal_seams(img: np.ndarray, seams: list[np.ndarray]) -> np.ndarray:
    img_with_seams = img.copy()
    for seam in seams:
        for col in range(len(seam)):
            row = seam[col]
            img_with_seams[row, col] = [0, 0, 255]  # red (BGR)
    return img_with_seams


# Locating the vertical seam with the least energy using dynamic programming.
def find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    rows, cols = energy.shape
    seam = np.zeros(rows, dtype=np.int32)
    seam_energy = energy.copy()

    for row in range(1, rows):
        for col in range(cols):
            min_energy = seam_energy[row - 1, col]
            if col > 0:
                min_energy = min(min_energy, seam_energy[row - 1, col - 1])
            if col < cols - 1:
                min_energy = min(min_energy, seam_energy[row - 1, col + 1])
            seam_energy[row, col] += min_energy

    seam[-1] = int(np.argmin(seam_energy[-1]))
    for row in range(rows - 2, -1, -1):
        prev_col = seam[row + 1]
        min_col = prev_col
        if prev_col > 0 and seam_energy[row, prev_col - 1] < seam_energy[row, min_col]:
            min_col = prev_col - 1
        if prev_col < cols - 1 and seam_energy[row, prev_col + 1] < seam_energy[row, min_col]:
            min_col = prev_col + 1
        seam[row] = min_col

    return seam


# Removing a vertical seam from the image. (Reduces width by 1.)
def remove_vertical_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    rows, cols, _ = img.shape
    img_removed = np.zeros((rows, cols - 1, 3), dtype=np.uint8)
    for row in range(rows):
        col = seam[row]
        img_removed[row, :col] = img[row, :col]
        img_removed[row, col:] = img[row, col + 1 :]
    return img_removed


def highlight_vertical_seams(img: np.ndarray, seams: list[np.ndarray]) -> np.ndarray:
    img_with_seams = img.copy()
    for seam in seams:
        for row in range(len(seam)):
            col = seam[row]
            img_with_seams[row, col] = [0, 0, 255]  # red (BGR)
    return img_with_seams


# Remove vertical seams from the image to reduce its width.
def compute_and_remove_vertical_seams(
    img: np.ndarray, num_seams: int, target_width: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    seams: list[np.ndarray] = []
    for _ in range(num_seams):
        energy = compute_energy(img)
        seam = find_vertical_seam(energy)
        seams.append(seam)
        img = remove_vertical_seam(img, seam)
        if img.shape[1] <= target_width:
            break
    return img, seams


# Remove horizontal seams from the image to reduce its height.
def compute_and_remove_horizontal_seams(
    img: np.ndarray, num_seams: int, target_height: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    seams: list[np.ndarray] = []
    for _ in range(num_seams):
        energy = compute_energy(img)
        seam = find_horizontal_seam(energy)
        seams.append(seam)
        img = remove_horizontal_seam(img, seam)
        if img.shape[0] <= target_height:
            break
    return img, seams


# Transport map for optimal seam removal order.
# Note: This implementation is extremely expensive as written (recomputes energy from scratch many times)
# but it is kept to match your original behavior.
def compute_transport_map(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    original_height, original_width = img.shape[:2]
    r, c = original_height - target_height, original_width - target_width

    T = np.full((r + 1, c + 1), np.inf)
    T[0, 0] = 0

    for i in range(r + 1):
        for j in range(c + 1):
            if i > 0:
                T[i, j] = min(
                    T[i, j],
                    T[i - 1, j]
                    + np.sum(
                        compute_energy(
                            remove_horizontal_seam(
                                img, find_horizontal_seam(compute_energy(img))
                            )
                        )
                    ),
                )
            if j > 0:
                T[i, j] = min(
                    T[i, j],
                    T[i, j - 1]
                    + np.sum(
                        compute_energy(
                            remove_vertical_seam(img, find_vertical_seam(compute_energy(img)))
                        )
                    ),
                )
    return T


def backtrack_seam_order(T: np.ndarray) -> list[str]:
    r, c = T.shape
    r, c = r - 1, c - 1
    seam_order: list[str] = []

    while r > 0 or c > 0:
        if r > 0 and (c == 0 or T[r - 1, c] < T[r, c - 1]):
            seam_order.append("horizontal")
            r -= 1
        else:
            seam_order.append("vertical")
            c -= 1
    return seam_order[::-1]


def resize_image_optimal(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    T = compute_transport_map(img, target_width, target_height)
    seam_order = backtrack_seam_order(T)

    for seam_type in seam_order:
        energy = compute_energy(img)
        if seam_type == "horizontal":
            seam = find_horizontal_seam(energy)
            img = remove_horizontal_seam(img, seam)
        else:
            seam = find_vertical_seam(energy)
            img = remove_vertical_seam(img, seam)

    return img


# Add a vertical seam by duplicating seam pixels
def add_vertical_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    rows, cols, _ = img.shape
    img_added = np.zeros((rows, cols + 1, 3), dtype=np.uint8)

    for row in range(rows):
        col = seam[row]
        for ch in range(3):
            img_added[row, :col, ch] = img[row, :col, ch]
            img_added[row, col, ch] = img[row, col, ch]
            img_added[row, col + 1 :, ch] = img[row, col:, ch]

    return img_added


# Add a horizontal seam by duplicating seam pixels
def add_horizontal_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    rows, cols, _ = img.shape
    img_added = np.zeros((rows + 1, cols, 3), dtype=np.uint8)

    for col in range(cols):
        row = seam[col]
        for ch in range(3):
            img_added[:row, col, ch] = img[:row, col, ch]
            img_added[row, col, ch] = img[row, col, ch]
            img_added[row + 1 :, col, ch] = img[row:, col, ch]

    return img_added


def compute_and_add_vertical_seams(img: np.ndarray, num_seams: int) -> np.ndarray:
    for _ in range(num_seams):
        energy = compute_energy(img)
        seam = find_vertical_seam(energy)
        img = add_vertical_seam(img, seam)
    return img


def compute_and_add_horizontal_seams(img: np.ndarray, num_seams: int) -> np.ndarray:
    for _ in range(num_seams):
        energy = compute_energy(img)
        seam = find_horizontal_seam(energy)
        img = add_horizontal_seam(img, seam)
    return img


def content_amplification(
    img: np.ndarray, scale_factor: float, target_width: int, target_height: int
) -> np.ndarray:
    scaled_width = int(img.shape[1] * scale_factor)
    scaled_height = int(img.shape[0] * scale_factor)
    scaled_img = cv2.resize(img, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    amplified_img = resize_image_optimal(scaled_img, target_width, target_height)
    return amplified_img


def remove_seam(img: np.ndarray, seam: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:  # vertical
        return np.array([np.delete(row, seam[i], axis=0) for i, row in enumerate(img)])
    if axis == 0:  # horizontal
        return np.delete(img, seam, axis=0)
    raise ValueError("axis must be 0 (horizontal) or 1 (vertical)")


def add_seam(img: np.ndarray, seam: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:  # vertical
        new_img = []
        for i, row in enumerate(img):
            seam_idx = seam[i]
            new_pixel = (
                np.mean(img[i, seam_idx - 1 : seam_idx + 1], axis=0)
                if 0 < seam_idx < img.shape[1] - 1
                else img[i, seam_idx]
            )
            new_row = np.insert(row, seam_idx, new_pixel, axis=0)
            new_img.append(new_row)
        return np.array(new_img)

    if axis == 0:  # horizontal
        new_img = img.copy()
        for j in range(img.shape[1]):
            seam_idx = seam[j]
            new_pixel = (
                np.mean(img[seam_idx - 1 : seam_idx + 1, j], axis=0)
                if 0 < seam_idx < img.shape[0] - 1
                else img[seam_idx, j]
            )
            new_img = np.insert(new_img, seam_idx, new_pixel, axis=0)
        return new_img

    raise ValueError("axis must be 0 (horizontal) or 1 (vertical)")


def retarget_image(img: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
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