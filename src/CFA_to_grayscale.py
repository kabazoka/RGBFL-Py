from PIL import Image

def to_grayscale(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to RGB mode
    image = image.convert("RGB")

    # Get the width and height of the image
    width, height = image.size

    # Process each pixel
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))

            # Find the maximum value among R, G, B channels
            max_value = max(r, g, b)

            # Set the rest of the channels to the maximum value
            r = max_value
            g = max_value
            b = max_value

            # Update the pixel with the new values
            image.putpixel((x, y), (r, g, b))

    return image

def resize_image(input_path, output_path, new_width, new_height):
    # Open the image file
    with Image.open(input_path) as img:
        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        
        # Save the resized image to the output path
        resized_img.save(output_path)


# Desired dimensions
new_width = 1872
new_height = 1404


# Call the function with the path to your image

input_image_path = 'output/CFA/Kodim22-resized-HT.bmp'
output_image_path = input_image_path.replace('.bmp', '-gray.bmp')
# input_image_path = 'output/image/out_kodim22-GM-32.png'
# output_image_path = 'output/CFA/out_kodim22-GM-32-resized.png'

# Convert the image to grayscale
grayimage = to_grayscale(input_image_path)
grayimage.save(output_image_path, mode="RGB")
# print("Image converted to grayscale successfully!")

# Resize the image
# resize_image(input_image_path, output_image_path, new_width, new_height)
# print(f"Image has been resized and saved to {output_image_path}")
