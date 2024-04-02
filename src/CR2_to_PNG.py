import rawpy
import imageio
import os

def convert_cr2_to_png(cr2_path, png_path):
    # Read the CR2 file
    with rawpy.imread(cr2_path) as raw:
        # Process the raw data into an image
        rgb = raw.postprocess()
    
    # Save the image as a PNG
    imageio.imsave(png_path, rgb)

# Example usage
cr2_folder = 'C:/Users/kabaz/Pictures/RGBFL/cr2'
png_folder = 'C:/Users/kabaz/Pictures/RGBFL/png'

# Run the function for all CR2 images in the folder
for image in os.listdir(cr2_folder):
    if image.endswith('.CR2'):
        cr2_path = os.path.join(cr2_folder, image)
        png_path = os.path.join(png_folder, image.replace('.CR2', '.png'))
        convert_cr2_to_png(cr2_path, png_path)
        print(f"Image {image} has been converted and saved to {png_path}")
