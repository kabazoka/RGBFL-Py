from PIL import Image

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
input_image_path = 'output/out_kodim22-GM-kd.jpg'

output_image_path = 'output/CFA/out_kodim22-GM-kd-resized.png'

# Resize the image
resize_image(input_image_path, output_image_path, new_width, new_height)
print(f"Image has been resized and saved to {output_image_path}")
