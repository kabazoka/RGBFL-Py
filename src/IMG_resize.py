from PIL import Image

def resize_image(input_path, output_path, new_width, new_height):
    # Open the image file
    with Image.open(input_path) as img:
        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        
        # Save the resized image to the output path
        resized_img.save(output_path)


for i in range(1, 25):
    kodim_num = f"{i:02d}"
    input_image_path = f'output/image/out_kodim{kodim_num}-GM.png'
    output_image_path = f'output/image/resized/out_kodim{kodim_num}-GM-resized.png'
    resize_image(input_image_path, output_image_path, 1872, 1404)
    print(f"Image {i} has been resized and saved to {output_image_path}")

# # Desired dimensions
# new_width = 1872
# new_height = 1404
# input_image_path = 'input/image/kodim22.png'
# output_image_path = input_image_path.replace('.png', '-resized.png')

# # Resize the image
# resize_image(input_image_path, output_image_path, new_width, new_height)
# print(f"Image has been resized and saved to {output_image_path}")
