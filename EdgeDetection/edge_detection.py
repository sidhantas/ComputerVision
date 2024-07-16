import numpy as np
from skimage import data, io
import matplotlib.pyplot as plt

def generate_gaussian_filter(ksize: int, sig=1):
    size = ksize * 2 + 1
    linspace = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    x, y = np.meshgrid(linspace, linspace)
    return np.multiply(1 / (2 * np.pi *np.square(sig)), np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sig)))

def convolution_filter(image: np.ndarray, kernel: np.ndarray, output_dtype=None) -> np.ndarray:
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    assert kernel_width == kernel_height, "kernel should be square"
    assert kernel_width % 2 == 1, "kernel dimensions should be odd"
    assert image_width >= kernel_width
    assert image_height >= kernel_height

    k = kernel_width // 2
    if output_dtype == None:
        output_dtype = image.dtype
    
    def convolution(image: np.ndarray, kernel: np.ndarray, y, x) -> int:
        convolution_val = 0
        for u in range(-k, k + 1):
            for v in range(-k, k + 1):
                convolution_val += image[y - u][x - v] * kernel[k + u][k + v]
            
        return convolution_val

    filtered_image = np.zeros(image.shape, dtype=output_dtype)
    for i in range(k, image_height - k):
        for j in range(k, image_width - k):
            filtered_image[i, j] = convolution(image, kernel, i, j)

    return filtered_image

def get_image_gradients(image: np.ndarray)-> (np.ndarray, np.ndarray):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = sobel_x.T
    gx = convolution_filter(image, sobel_x, np.float64)
    gy = convolution_filter(image, sobel_y, np.float64)

    edge_angle = np.arctan(np.divide(gy, gx))
    np.nan_to_num(edge_angle, copy=False)
    return np.sqrt(np.square(gx) + np.square(gy)), edge_angle, gx, gy


def non_maximum_suppression(edge_gradient: np.ndarray, edge_angle: np.ndarray)-> np.ndarray:
    def get_angle_offset(angle: float):
        if angle <= np.pi / 2 and angle > np.pi / 3:
            return 1, 0
        if angle <= np.pi / 3 and angle > np.pi / 6:
            return 1, 1
        if angle <= np.pi / 6 and angle > -np.pi / 6:
            return 0, 1
        if angle <= -np.pi / 6 and angle > -np.pi / 3:
            return -1, 1
        if angle <= -np.pi / 3 and angle >= -np.pi / 2:
            return -1, 0

        raise ValueError("Invalid Angle", angle)

    height, width = edge_gradient.shape
    non_maximum_suppressed_image = np.zeros(edge_gradient.shape, dtype=edge_gradient.dtype)
    for i in range(3, height - 3):
        for j in range(3, width - 3):
            offset_y, offset_x = get_angle_offset(edge_angle[i, j])

            comparison_pixel_left = edge_gradient[i - offset_y, j - offset_x]
            current_pixel = edge_gradient[i, j]
            comparison_pixel_right =edge_gradient[i + offset_y, j + offset_x]
            if comparison_pixel_left <= current_pixel >= comparison_pixel_right:
                non_maximum_suppressed_image[i, j] = current_pixel

    return non_maximum_suppressed_image

def hysteresis_thresholding(min_val, max_val, image: np.ndarray):
    height, width = image.shape
    thresholded_image = np.zeros(image.shape, dtype=np.uint8)
    possible_edges = []
    for i in range(height):
        for j in range(width):
            if image[i, j] >= max_val:
                thresholded_image[i, j] = 255
            if min_val < image[i, j] < max_val:
                possible_edges.append((i, j))

    added_edge = True

    def has_adjacent_edge(i, j):
        for u in range(-1, 2):
            for v in range(-1, 2):
                if thresholded_image[i + u, j + v] == 255:
                    return True

        return False

    while added_edge:
        added_edge = False
        new_possible_edges = []
        for (i, j) in possible_edges:
            if has_adjacent_edge(i, j):
                thresholded_image[i, j] = 255
                added_edge = True
                possible_edges
            else:
                new_possible_edges.append((i, j))

        possible_edges = new_possible_edges

    return thresholded_image

def canny(image: np.ndarray) -> np.ndarray:
    # 1. Gaussian Blurring so the edge detection is less susceptible to noise
    filtered_image = convolution_filter(image, generate_gaussian_filter(2))

    # 2. Create 2 Image gradients Gx and Gy
    edge_gradient, edge_angle, gx, gy = get_image_gradients(filtered_image)

    # 3. Non-maximum supression using angle information
    non_maximum_suppressed_image = non_maximum_suppression(edge_gradient, edge_angle)

    # 4. Apply hysteresis thresholding
    thresholded_image = hysteresis_thresholding(50, 150, non_maximum_suppressed_image)
    return thresholded_image


if __name__ == '__main__':
    # Canny Edge Detector Steps
    image = data.camera()

    edges = canny(image)
    io.imsave("img.png", image)
    io.imsave("Edges.png", edges)
    io.imshow(edges)
    io.show()

