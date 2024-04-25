import cv2
import numpy as np
import colour
from tqdm import tqdm
import logging
import multiprocessing
from multiprocessing import Pool

# Utility functions
def RGB_to_XYZ(RGB):
    # Convert the RGB values to XYZ values
    XYZ = colour.RGB_to_XYZ(RGB, "sRGB", None, None)
    x, y, z = XYZ
    XYZ = (x / 10, y / 10, z / 10)

    return XYZ

def BGR_to_LAB(bgr):
    """
    Convert a BGR color to Lab color space using the colour-science package.

    Parameters:
    - bgr: A (3,) numpy array or a tuple/list of BGR values in the range 0-255.

    Returns:
    - lab: A (3,) numpy array of Lab values.
    """
    # Convert BGR to RGB
    rgb = bgr[::-1]  # This reverses the BGR to RGB since BGR is just RGB in reverse order
    # Normalize RGB values to the range [0, 1] for the conversion
    rgb_normalized = np.array(rgb) / 255.0
    # Convert RGB to XYZ using the sRGB color space as the source space
    xyz = RGB_to_XYZ(rgb_normalized)
    # Convert XYZ to Lab
    lab = colour.XYZ_to_Lab(xyz)
    
    return lab

# Node and KDTree classes
class Node:
    def __init__(self, point=None, index=None, left=None, right=None):
        self.point = point
        self.index = index
        self.left = left
        self.right = right

class KDTree:
    def build_tree(self, points, depth=0):
        if not points:
            return None
        
        # Assuming points as [(color, index)]
        k = len(points[0][0])  # Correctly access the dimensions of the color
        axis = depth % k

        points.sort(key=lambda x: x[0][axis])
        median = len(points) // 2

        return Node(
            point=points[median][0],
            index=points[median][1],
            left=self.build_tree(points[:median], depth + 1),
            right=self.build_tree(points[median+1:], depth + 1)
        )
    
    def construct_tree(self, points):
        # Adjusting to pass points as [(color, index)]
        indexed_points = [(point, i) for i, point in enumerate(points)]
        self.root = self.build_tree(indexed_points)

def calculate_nearest_for_color(args):
    pixel, kd_tree = args
    pixel_lab = BGR_to_LAB(pixel)
    index, _ = nearest_neighbor(kd_tree, pixel_lab)
    b_translated = min(max(((index // (65*65)) * 4) - 1, 0), 255)
    g_translated = min(max((((index % (65*65)) // 65) * 4) - 1, 0), 255)
    r_translated = min(max(((index % 65) * 4) - 1, 0), 255)

    return pixel, (b_translated, g_translated, r_translated)

def find_nearest(node, target, depth=0, best=None):
    if node is None:
        return best

    k = len(target)
    axis = depth % k
    next_best = None
    opposite_branch = None

    # Determine closer side
    if target[axis] < node.point[axis]:
        next_node = node.left
        opposite_branch = node.right
    else:
        next_node = node.right
        opposite_branch = node.left

    # Explore the side closer to the target first
    next_best = find_nearest(next_node, target, depth + 1, best)

    # Now, check if we need to explore the other side
    if next_best is None or colour.delta_E(target, next_best.point, 'CIE 1976') > abs(target[axis] - node.point[axis]):
        next_best = find_nearest(opposite_branch, target, depth + 1, next_best or best)

    # Update the best node if the current node is closer to the target
    if next_best is None or colour.delta_E(target, next_best.point) > colour.delta_E(target, node.point):
        next_best = node

    return next_best

def nearest_neighbor(kd_tree, target):
    node = find_nearest(kd_tree.root, target)
    return node.index, node.point

def extract_unique_colors(image):
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    return [tuple(color) for color in unique_colors]

def apply_lut_with_cache(image, cache):
    height, width = image.shape[:2]
    for y in tqdm(range(height), leave=False):
        for x in range(width):
            color = tuple(image[y, x])
            image[y, x] = cache[color]
            # if y == 0 and x == 38:
            #     logging.info(f"Color: {color}, Translated Color: {cache[color]}")

def update_cache_with_multiprocessing(unique_colors, kd_tree):
    cache = {}
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        logging.info(f"Number of processes: {multiprocessing.cpu_count()}")
        results = pool.map(calculate_nearest_for_color, [(color, kd_tree) for color in unique_colors])
    for color, translated_color in results:
        cache[color] = translated_color
    return cache

def convert_bgr_to_lab_chunk(bgr_values_chunk):
    lab_values_chunk = []
    for bgr in bgr_values_chunk:
        # Normalize BGR values to [0, 1]
        bgr_normalized = np.array(bgr) / 255.0
        # Convert BGR to RGB
        rgb_normalized = bgr_normalized[::-1]
        # Convert RGB to XYZ
        xyz = RGB_to_XYZ(rgb_normalized)
        xyz = np.array(xyz)
        # Convert XYZ to Lab
        lab = colour.XYZ_to_Lab(xyz)
        lab_values_chunk.append(lab)
    return lab_values_chunk

def read_lut_and_convert_to_lab_parallel(filename):
    # Load the LUT from the file
    with open(filename, 'r') as file:
        bgr_lut = [list(map(float, line.strip().split())) for line in file]

    # Number of processes to use; often set to the number of CPUs available
    num_processes = multiprocessing.cpu_count()

    # Create chunks of the BGR LUT for each process
    chunk_size = len(bgr_lut) // num_processes
    bgr_lut_chunks = [bgr_lut[i:i + chunk_size] for i in range(0, len(bgr_lut), chunk_size)]

    # Convert BGR LUT to LAB LUT in parallel
    with Pool(num_processes) as pool:
        lab_lut_chunks = pool.map(convert_bgr_to_lab_chunk, bgr_lut_chunks)

    # Flatten the list of lists into a single list
    lab_lut = [lab for chunk in lab_lut_chunks for lab in chunk]

    return lab_lut


if __name__ == "__main__":
    for i in range(1, 25):
        kodim_num = f"{i:02d}"
        image_filename = f'input/image/out_kodim{kodim_num}.png'
        output_img_filename = f'output/image/out_kodim{kodim_num}-GM.png'
        lut_filename = f'input/LUT/kodim{kodim_num}.txt'

        logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info("Reading LUT and converting to LAB...")
        lut_lab = read_lut_and_convert_to_lab_parallel(lut_filename)

        logging.info("Constructing KD-Tree...")
        kd_tree = KDTree()
        kd_tree.construct_tree(lut_lab)

        logging.info("Extracting unique colors from image...")
        image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        unique_colors = extract_unique_colors(image)

        logging.info("Constructing KD-Tree...")
        kd_tree = KDTree()
        kd_tree.construct_tree(lut_lab)

        logging.info("Calculating nearest neighbors for unique colors and updating cache with multiprocessing...")
        cache = update_cache_with_multiprocessing(unique_colors, kd_tree)

        logging.info("Applying LUT to image using cache...")
        apply_lut_with_cache(image, cache)

        cv2.imwrite(output_img_filename, image)
        logging.info("Done applying LUT to image.")

    # kodim_num = "01"
    # image_filename = f'input/image/out_kodim{kodim_num}.png'
    # output_img_filename = f'output/image/out_kodim{kodim_num}-GM.png'
    # lut_filename = f'input/LUT/kodim{kodim_num}.txt'

    # logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # logging.info("Reading LUT and converting to LAB...")
    # lut_lab = read_lut_and_convert_to_lab_parallel(lut_filename)

    # logging.info("Constructing KD-Tree...")
    # kd_tree = KDTree()
    # kd_tree.construct_tree(lut_lab)

    # logging.info("Extracting unique colors from image...")
    # image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    # unique_colors = extract_unique_colors(image)

    # logging.info("Constructing KD-Tree...")
    # kd_tree = KDTree()
    # kd_tree.construct_tree(lut_lab)

    # logging.info("Calculating nearest neighbors for unique colors and updating cache with multiprocessing...")
    # cache = update_cache_with_multiprocessing(unique_colors, kd_tree)

    # logging.info("Applying LUT to image using cache...")
    # apply_lut_with_cache(image, cache)

    # cv2.imwrite(output_img_filename, image)
    # logging.info("Done applying LUT to image.")
