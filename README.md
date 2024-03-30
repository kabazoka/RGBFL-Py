# README.md

## Overview

This repository contains two Python scripts designed for image processing and color transformation tasks. The scripts utilize advanced color science techniques to analyze and modify images in various color spaces.

### Scripts

- `RGBFL_FL.py`: This script focuses on reading color measurement data, interpolating colors, and visualizing the relationship between image colors and lighting conditions to optimize color accuracy.
- `RGBFL_GM.py`: A tool for converting images using a color lookup table (LUT) and nearest neighbor interpolation within the LAB color space, aiming to enhance image quality through precise color adjustments.

## Prerequisites

Before running the scripts, ensure you have the following dependencies installed:

- Python 3.6 or newer
- OpenCV (`cv2`)
- NumPy
- Pandas
- Matplotlib
- Scipy
- colour-science
- tqdm
- logging
- multiprocessing

You can install these dependencies using pip:

```bash
pip install numpy opencv-python pandas matplotlib scipy colour-science tqdm
```

## Usage

### RGBFL_FL.py

1. Ensure you have an Excel file with color measurement data and an image file you want to process.  
2. Modify the `image_path` and `data_file_path` variables in the script to point to your files.
3. Run the script:

```bash
python RGBFL_FL.py
```

This will generate output that includes interpolated color values and visualization plots comparing the image and lighting condition color gamuts.

### RGBFL_GM.py

1. Prepare a LUT file and an image file for processing.
2. Set the `image_filename` and `lut_filename` variables in the script to your specific files.
3. Execute the script:

```bash
python RGBFL_GM.py
```

The script will apply the LUT to the image using multiprocessing for efficient color translation and save the output image.

## Note

- The scripts generate logs and output images in specified directories; ensure these exist or modify the script to your directory structure.
- You might need to adjust file paths and parameters based on your project's requirements.

## Contribution

Feel free to fork this repository and submit pull requests to contribute to the project. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)