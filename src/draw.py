from PIL import Image

# Create a new 5x5 image with red pixels
image = Image.new("RGB", (1, 1), "green")

# Save the image as a BMP file
image.save("output/green_pixel.jpg")