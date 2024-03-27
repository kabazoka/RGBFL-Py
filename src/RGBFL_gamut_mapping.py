import cv2
import numpy as np
import colour
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
import logging

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


def get_lut_index_with_min_delta_e(target_b, target_g, target_r, lut):
    """Find the index of the LUT entry with the minimum Delta E from the target BGR."""
    min_delta_e = float('inf')
    closest_index = -1
    target_bgr = (target_b, target_g, target_r)
    target_lab = bgr_to_lab_colour_science(target_bgr)

    for index in range(len(lut)):
        
        lab = lut[index]

        LAB_target = (target_lab[0], target_lab[1], target_lab[2])
        LAB_in_LUT = (lab[0], lab[1], lab[2])
        # print(f"LAB_target: {LAB_target}, LAB_in_LUT: {LAB_in_LUT}")
        current_delta_e = colour.delta_E(LAB_target, LAB_in_LUT, method='CIE 2000')
        if current_delta_e < min_delta_e:
            min_delta_e = current_delta_e
            closest_index = index

    target_bgr = (target_b, target_g, target_r)
    lut_lab = lut[closest_index]

    # print(f"Target BGR: {target_bgr}, Target lab: {target_lab}, LUT Lab: {lut_lab}")
    # print(f"Min Delta E: {min_delta_e}")

    return closest_index

def apply_lut_section(args):
    y_start, y_end, width, image, lut = args
    # Create a copy or slice of the image for the section to avoid modifying the original image directly
    image_section = image[y_start:y_end].copy()

    for y in range(y_end - y_start):  # Adjust y to start from 0 within the section
        for x in range(width):
            b, g, r = image_section[y, x]
            closest_index = get_lut_index_with_min_delta_e(b, g, r, lut)
            print(f"Closest index: {closest_index}")
            b_translated = (closest_index // (16*16)) * 16
            g_translated = (closest_index % (16*16)) // 16 * 16
            r_translated = (closest_index % (16*16) % 16) * 16
            image_section[y, x] = [b_translated, g_translated, r_translated]
    return y_start, image_section  # Return the starting y and the processed section


def main(image_filename, lut, output_filename):
    image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    num_processes = cpu_count()
    pool = Pool(processes=num_processes)
    logging.info(f"Using {num_processes} processes")

    step = height // num_processes
    tasks = [(i, min(i + step, height), width, image, lut) for i in range(0, height, step)]

    results = pool.map(apply_lut_section, tasks)

    for y_start, section in results:
        image[y_start:y_start+section.shape[0], :] = section  # Correctly replace the section

    cv2.imwrite(output_filename, image)


if __name__ == "__main__":
    image_filename = 'input/image/out_kodim22_ds.png'
    output_img_filename = 'output/image/out_kodim22_ds-GM.png'
    # lut_filename = 'input/out_gmlut_name.txt'
    downsample_lut_filename = 'input/LUT/kodim22_4096.txt'

    # downsampled_lut = downsample_lut(read_lut_from_file(lut_filename))
    # write_lut_to_file(downsampled_lut, downsample_lut_filename)
    downsampled_lut_lab = read_lut_and_convert_to_lab(downsample_lut_filename)

    # Set up logging configuration
    logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # write_lut_to_file(downsampled_lut_lab, 'python/output/downsampled_Lab.txt')
    logging.info("Starting main function")
    main(image_filename, downsampled_lut_lab, output_img_filename)
    logging.info("Finished main function")