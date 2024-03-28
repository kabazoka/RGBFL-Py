import cv2
import numpy as np
import colour
from tqdm import tqdm
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


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

i = 0
    

def find_nearest(node, target, depth=0, best=None):

    # global i
    # i += 1
    # print(f"Depth: {depth}")
    # print(f"i: {i}")
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
    if next_best is None or colour.delta_E(target, next_best.point) > abs(target[axis] - node.point[axis]):
        next_best = find_nearest(opposite_branch, target, depth + 1, next_best or best)

    # Update the best node if the current node is closer to the target
    if next_best is None or colour.delta_E(target, next_best.point) > colour.delta_E(target, node.point):
        next_best = node

    return next_best


def nearest_neighbor(kd_tree, target):
    node = find_nearest(kd_tree.root, target)
    return node.index, node.point

# Load your LUT from the file
def load_lut(filename):
    with open(filename, 'r') as file:
        points = [tuple(map(float, line.strip().split())) for line in file]
    return points

def RGB_to_XYZ_Colour_Science(RGB):
    # Convert the RGB values to XYZ values
    XYZ = colour.RGB_to_XYZ(RGB, "sRGB", None, "CAT02")
    x, y, z = XYZ
    XYZ = (x / 10, y / 10, z / 10)

    return XYZ

def bgr_to_lab_colour_science(bgr):
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
    xyz = RGB_to_XYZ_Colour_Science(rgb_normalized)
    
    # Convert XYZ to Lab
    lab = colour.XYZ_to_Lab(xyz)
    
    return lab


def read_lut_from_file(filename):
    """Read the LUT from a file and return it as a list of tuples in BGR format."""
    lut = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                b, g, r = map(float, parts)
                lut.append((b, g, r))
    return lut

def read_lut_and_convert_to_lab(filename):
    """
    Read a LUT from a text file with BGR values in the range [0, 255], 
    and convert it to LAB color space.

    Parameters:
    - filename: Path to the LUT file.

    Returns:
    - A list of LAB color space values.
    """
    # Load the LUT from the file
    with open(filename, 'r') as file:
        bgr_lut = [list(map(float, line.strip().split())) for line in file]

    # Convert the BGR LUT to LAB LUT
    lab_lut = []
    for bgr in bgr_lut:
        # Normalize BGR values to [0, 1]
        bgr_normalized = np.array(bgr) / 255.0

        # Convert BGR to RGB
        rgb_normalized = bgr_normalized[::-1]

        # print(f"RGB: {rgb_normalized}")

        # Convert RGB to XYZ
        xyz = RGB_to_XYZ_Colour_Science(rgb_normalized)

        # print(f"XYZ: {xyz}")

        xyz = np.array(xyz)
        
        # Convert XYZ to Lab
        lab = colour.XYZ_to_Lab(xyz)

        # print(f"LAB: {lab}")
        lab_lut.append(lab)

    return lab_lut



def downsample_lut(original_lut, original_size=65, target_size=16):
    """
    Downsample a 65x65x65 LUT to a 16x16x16 LUT by picking specific values.

    Args:
    - original_lut: The original LUT as a list of RGB tuples.
    - original_size: The size of each dimension in the original LUT.
    - target_size: The size of each dimension in the target (downsampled) LUT.

    Returns:
    - A downsampled LUT as a list of RGB tuples.
    """
    downsampled_lut = []
    step = original_size / target_size

    for b in range(target_size):
        for g in range(target_size):
            for r in range(target_size):
                # Calculate the indices in the original LUT to pick values from
                orig_b = int(b * step)
                orig_g = int(g * step)
                orig_r = int(r * step)

                # Ensure we don't exceed original LUT bounds
                orig_b = min(orig_b, original_size - 1)
                orig_g = min(orig_g, original_size - 1)
                orig_r = min(orig_r, original_size - 1)

                # Calculate the linear index in the original LUT
                index = orig_b * original_size**2 + orig_g * original_size + orig_r
                downsampled_lut.append(original_lut[index])

    return downsampled_lut

def write_lut_to_file(lut, filename):
    """Write the LUT to a file."""
    with open(filename, 'w') as file:
        for rgb in lut:
            file.write(f"{rgb[0]} {rgb[1]} {rgb[2]}\n")




def main(image_filename, output_img_filename):
    """Apply LUT to each pixel of the image using a custom index translation method for a 16x16x16 LUT."""
    # Load the image
    image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    height, width = image.shape[:2]

    lut_points = load_lut('input/LUT/downsampled_Lab.txt')  # Load the LUT
    kd_tree = KDTree()
    kd_tree.construct_tree(lut_points)

    for y in tqdm(range(0, height, 1), leave=False):
        for x in tqdm(range(0, width, 1), leave=False):
            b, g, r = image[y, x]  # Read pixel assuming the image is in BGR format
            pixel_lab = bgr_to_lab_colour_science((b, g, r))

            # print(f"Pixel ({x}, {y}): ({b}, {g}, {r}) -> ({pixel_lab[0]}, {pixel_lab[1]}, {pixel_lab[2]})")
            index, closest_color = nearest_neighbor(kd_tree, pixel_lab)
            
            # print(f"Closest index: {index}")
            # Correctly translate the index back into RGB values
            b_translated = (index // (16*16)) * 16
            g_translated = (index % (16*16)) // 16 * 16
            r_translated = (index % (16*16) % 16) * 16

            # Assign the translated RGB values to the pixel
            image[y, x] = [b_translated, g_translated, r_translated]

            # print(f"Pixel ({x}, {y}): ({b}, {g}, {r}) -> ({b_translated}, {g_translated}, {r_translated})")
            # print(f"Closest index: {closest_index}")

    # Save the modified image
    cv2.imwrite(output_img_filename, image)

# Example usage
image_filename = 'input/image/out_kodim22.png'
output_img_filename = 'output/out_kodim22-GM-kd.jpg'
downsample_lut_filename = 'input/LUT/downsampled_Lab.txt'

# lut = read_lut_from_file(lut_filename)
# downsample_lut = downsample_lut(lut)
# write_lut_to_file(downsample_lut, downsample_lut_filename)

main(image_filename, output_img_filename)