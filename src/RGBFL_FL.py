import cv2
import numpy as np
import tqdm
import colour
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator
import logging
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from multiprocessing import Pool, cpu_count
from scipy.spatial import KDTree

# Function to convert BGR to CIELab color space
def bgr_to_lab(bgr):
    bgr = np.uint8([[bgr]])
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[0][0]

def read_color_measurement_data_combined(file_path):
    # Deal with the FL and RGB data
	FL_red = []
	FL_green = []
	FL_blue = []
	color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'black']
	
	df_all = pd.read_excel(file_path)
	for i in range(0, 27, 1):
		df_FL_R = df_all.at[i, "FL_R"]
		df_FL_G = df_all.at[i, "FL_G"]
		df_FL_B = df_all.at[i, "FL_B"]
		FL_red.append(df_FL_R)
		FL_green.append(df_FL_G)
		FL_blue.append(df_FL_B)
	
	FL_pts_all = np.vstack([FL_red, FL_green, FL_blue]).T

	return FL_pts_all

def read_color_measurement_data_separate(file_path):
	# Deal with the FL and RGB data
	FL_red = []
	FL_green = []
	FL_blue = []
	
	df_all = pd.read_excel(file_path)
	for i in range(0, 27, 1):
		df_FL_R = df_all.at[i, "FL_R"]
		df_FL_G = df_all.at[i, "FL_G"]
		df_FL_B = df_all.at[i, "FL_B"]
		FL_red.append(df_FL_R)
		FL_green.append(df_FL_G)
		FL_blue.append(df_FL_B)

	X = []
	Y = []
	Z = []
	for i in range(0, 8, 1):
		for j in range(0, 27, 1):
			df_X = df_all.at[j, "X{}".format(i + 1)]
			df_Y = df_all.at[j, "Y{}".format(i + 1)]
			df_Z = df_all.at[j, "Z{}".format(i + 1)]
            # Scale the XYZ values to 0-1 by dividing 255
			X.append(df_X / 255)
			Y.append(df_Y / 255)
			Z.append(df_Z / 255)
	
	data_pts = np.vstack([X, Y, Z]).T

	# Separate the data points into 8 colors
	color_pts = np.split(data_pts, 8)
	color_names = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'black']
	color_dict = dict(zip(color_names, color_pts))

	return color_dict

def read_image(image_path):
	# Input the image
	lab_colors = []
	image = cv2.imread(image_path)
	height, width = image.shape[:2]

	# Convert each pixel's color to CIELab
	# Downsampling the image to speed up the process (100 * 100 blocks)
	for y in range(0, height, 1):
		for x in range(0, width, 1):
			bgr = image[y, x]
			lab = bgr_to_lab(bgr)
			lab_colors.append(lab)

	# Convert list to NumPy array for easier manipulation
	lab_colors = np.array(lab_colors)

	return lab_colors

def interpolate_color(tri, values, query_point):
    interpolator_x = LinearNDInterpolator(tri, values[:, 0])
    interpolator_y = LinearNDInterpolator(tri, values[:, 1])
    interpolator_z = LinearNDInterpolator(tri, values[:, 2])
    x = interpolator_x(query_point)
    y = interpolator_y(query_point)
    z = interpolator_z(query_point)
    return (x, y, z)

def plot_interpolating_FL_combinations(allFLs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for allFL in allFLs:
        ax.scatter(allFL[0], allFL[1], allFL[2], s=1, c='red')
    ax.set_xlabel('FL_R')
    ax.set_ylabel('FL_G')
    ax.set_zlabel('FL_B')
    plt.show()

def plot_image_and_FL_gamut(pixels_lab, primaries_best_FL, dominant_color, selected_pixels_lab, all_FLs, G_FL, R_FL, B_FL):
    # Create a 3D plot
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    
    # Calculate the convex hull of the LAB colors
    image_hull = ConvexHull(pixels_lab)
    # Calculate the convex hull of the FL colors
    FL_hull = ConvexHull(primaries_best_FL)
    r_hull = ConvexHull(R_FL)
    g_hull = ConvexHull(G_FL)
    b_hull = ConvexHull(B_FL)
    # # Calculate the convex hull of all FL colors
    # all_FLs = np.array(all_FLs)
    # all_FLs_hull = ConvexHull(all_FLs)


    # Plot the vertices of the image color gamut convex hull
    for simplex in image_hull.simplices:
        ax.plot(pixels_lab[simplex, 1], pixels_lab[simplex, 2], pixels_lab[simplex, 0], color='orange', linewidth=1.2)
    
    # Scatter the colors of the image
    ax.scatter(pixels_lab[:,1], pixels_lab[:,2], pixels_lab[:,0], s=0.01, c='orange')
    
    # Convert the colors to LAB color space
    lab_FL_colors = []
    lab_FL_R = []
    lab_FL_G = []
    lab_FL_B = []
    
    for color in primaries_best_FL:
        lab = colour.XYZ_to_Lab(color)
        lab_FL_colors.append(lab)

    for color in R_FL:
        lab = colour.XYZ_to_Lab(color)
        lab_FL_R.append(lab)

    for color in G_FL:
        lab = colour.XYZ_to_Lab(color)
        lab_FL_G.append(lab)

    for color in B_FL:
        lab = colour.XYZ_to_Lab(color)
        lab_FL_B.append(lab)

    lab_FL_colors = np.array(lab_FL_colors)  # Convert list to NumPy array
    lab_FL_R = np.array(lab_FL_R)
    lab_FL_G = np.array(lab_FL_G)
    lab_FL_B = np.array(lab_FL_B)

    # # Convert the colors to LAB color space of all FLs
    # lab_all_FLs = []
    # for color in all_FLs:
    #     lab = colour.XYZ_to_Lab(color)
    #     lab_all_FLs.append(lab)

    # lab_all_FLs = np.array(lab_all_FLs)

    # Plot the vertices of the FL color gamut convex hull
    # for simplex in all_FLs_hull.simplices:
    #     ax.plot(lab_all_FLs[simplex, 1], lab_all_FLs[simplex, 2], lab_all_FLs[simplex, 0], color='blue', linewidth=0.8, label='All FL gamut')

    # # Scatter the colors of FL gamut and label them
    # color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'black']
    # for i, color in enumerate(lab_FL_colors):
    #     ax.scatter(color[1], color[2], color[0], s=10, label=color_list[i], c=color_list[i])
    #     ax.text(color[1], color[2], color[0], color_list[i])
    
    # # Plot the vertices of the FL color gamut convex hull
    # for simplex in FL_hull.simplices:
    #     ax.plot(lab_FL_colors[simplex, 1], lab_FL_colors[simplex, 2], lab_FL_colors[simplex, 0], color='red', linewidth=0.8, label='FL gamut')

    # # Plot the dominant color
    # ax.scatter(dominant_color[1], dominant_color[2], dominant_color[0], s=50, c='green', label='Dominant color')

    # # Plot the selected pixels
    # ax.scatter(selected_pixels_lab[:, 1], selected_pixels_lab[:, 2], selected_pixels_lab[:, 0], s=10, c='blue', label='Selected pixels')

    
    # Set the axis thicker
    ax.xaxis.set_tick_params(width=4)
    ax.yaxis.set_tick_params(width=4)
    ax.zaxis.set_tick_params(width=4)

    # Set the label of the axis thicker
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.zaxis.label.set_size(15)
    ax.xaxis.label.set_weight('bold')
    ax.yaxis.label.set_weight('bold')
    ax.zaxis.label.set_weight('bold')

    # Set the background color of the plot
    bgclr = (0.9, 0.9, 0.9)
    ax.xaxis._axinfo['grid'].update(color = bgclr)
    ax.yaxis._axinfo['grid'].update(color = bgclr)
    ax.zaxis._axinfo['grid'].update(color = bgclr)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Set the axis line width
    ax.xaxis.line.set_lw(4)
    ax.yaxis.line.set_lw(4)
    ax.zaxis.line.set_lw(4)

    ax.tick_params(axis='both', which='major', labelsize=20, width=3)
    ax.tick_params(axis='both', which='minor', labelsize=15, width=2)

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # set the name of the axis
    ax.set_xlabel('a*', fontsize=20, fontweight='bold', labelpad=20)
    ax.set_ylabel('b*', fontsize=20, fontweight='bold', labelpad=20)
    ax.set_zlabel('L*', fontsize=20, fontweight='bold', labelpad=20)

    #

    plt.show()


def interpolate_color_for_color(FL, tri, values):
    return interpolate_color(tri, values, FL)

def compute_loss_for_FL(interpolated_FL_per_color, dominant_color, FL):
    target_XYZ = np.array(interpolated_FL_per_color[FL])
    target_lab = colour.XYZ_to_Lab(target_XYZ)
    loss = colour.delta_E(dominant_color, target_lab)
    return FL, loss

def find_best_FL(all_interpolated_FL, dominant_color):
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(cpu_count)

    color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'black']
    tasks = []
    for color in color_list:
        interpolated_FL_per_color = all_interpolated_FL[color]
        for FL in interpolated_FL_per_color.keys():
            # Add the task for computing the loss of each FL setting
            tasks.append((interpolated_FL_per_color, dominant_color, FL))

    # Step 2: Use pool.starmap to apply the worker function across all tasks
    results = pool.starmap(compute_loss_for_FL, tasks)

    # Close the pool and wait for all tasks to complete
    pool.close()
    pool.join()

    # Step 3: Process the results to find the FL setting with the lowest loss
    lowest_loss = float('inf')
    best_FL = None
    for FL, loss in results:
        if loss < lowest_loss:
            lowest_loss = loss
            best_FL = FL

    return best_FL, lowest_loss

def find_FL_with_mean(image_path, data_file_path):
    # Log the processing image path and the name of the excel file
    logging.info(f"Processing image: {image_path}")
    logging.info(f"Excel file: {data_file_path}")

    # Read the color measurement data from the file
    front_lights = read_color_measurement_data_combined(data_file_path)
    measured_colors = read_color_measurement_data_separate(data_file_path)

    # Prepare Delaunay triangulation
    tri = Delaunay(front_lights)

    # Convert the entire image to Lab color space at once
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image.astype("float32") / 255
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    all_pixels_lab = image_lab.reshape(-1, 3)

    # Downsample the image to 1/4 of the original size
    height, width = image.shape[:2]
    downsampled_pixels_lab = []
    for i in range(0, height * width // 4, 4):
        max_L = max(all_pixels_lab[i][0], all_pixels_lab[i + 1][0], all_pixels_lab[i + 2][0], all_pixels_lab[i + 3][0]) / 2
        max_a = max(all_pixels_lab[i][1], all_pixels_lab[i + 1][1], all_pixels_lab[i + 2][1], all_pixels_lab[i + 3][1])
        max_b = max(all_pixels_lab[i][2], all_pixels_lab[i + 1][2], all_pixels_lab[i + 2][2], all_pixels_lab[i + 3][2])
        downsampled_pixels_lab.append((max_L, max_a, max_b))

    logging.info(f"Size of downsampled pixels: {len(downsampled_pixels_lab)}")

    # Sort the downsampled pixels by absolute L*, a*, b* values
    downsampled_pixels_lab = np.array(downsampled_pixels_lab)

    # Calculate the Euclidean distance from (0, 0, 0)
    distances = np.sqrt(np.sum(downsampled_pixels_lab**2, axis=1))

    # Sort the array based on the calculated distances
    sorted_indices = np.argsort(distances)  # This gives indices that would sort the array
    sorted_pixels_lab = downsampled_pixels_lab[sorted_indices]

    # Select the 10% of the pixels with the highest distances
    top_10_percent_idx = int(len(sorted_pixels_lab) * 0.9)  # Index to start the top 10%
    selected_pixels_lab = sorted_pixels_lab[top_10_percent_idx:]

    # logging.info(f"Size of selected pixels after sorting by distance: {len(selected_pixels_lab)}")
    
    # Choose a color as the color representing the image
    # By finding the mean of each L* a* b* channel of every pixels in the image
    # mean_L = np.mean(selected_pixels_lab[:, 0])
    # mean_a = np.mean(selected_pixels_lab[:, 1])
    # mean_b = np.mean(selected_pixels_lab[:, 2])
    # dominant_color = (mean_L, mean_a, mean_b)
    # logging.info(f"Mean L: {mean_L}, Mean a: {mean_a}, Mean b: {mean_b}")

    # # Finding the centroid of the image gamut found by the convex hull
    image_hull = ConvexHull(selected_pixels_lab)
    image_centroid = np.mean(image_hull.points[image_hull.vertices], axis=0)
    logging.info(f"Centroid of the image gamut: {image_centroid}")

    # Choose the centroid of the image gamut as the dominant color
    dominant_color = (image_centroid[0], image_centroid[1], image_centroid[2])

    print(dominant_color[0], dominant_color[1], dominant_color[2])

    # Generate all possible predicted colors just once
    color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'black']
    # Use a pool of workers to parallelize the generation of predicted colors
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)

    logging.info("Generating all possible predicted colors")
    all_interpolated_FL = {}
    for color in color_list:
        FL_ranges = [(FL_R, FL_G, FL_B) for FL_R in range(0, 256, 5) for FL_G in range(0, 256, 5) for FL_B in range(0, 256, 5) if FL_R == 255 or FL_G == 255 or FL_B == 255]
        func = partial(interpolate_color_for_color, tri=tri, values=measured_colors[color])
        results = pool.map(func, FL_ranges)
        all_interpolated_FL[color] = dict(zip(FL_ranges, results))

    pool.close()
    pool.join()

    # # Draw a 3d plot of the front lights combination
    # allFLs = all_interpolated_FL[color_list[0]]
    # plot_interpolating_FL_combinations(allFLs)

    logging.info(f"Size of interpolated_FL_per_color: {len(all_interpolated_FL[color_list[0]])}")

    # Record all of the primaries under every FL setting and convert them to Lab color space
    all_FLs = []
    for color in color_list:
        for FL in all_interpolated_FL[color].keys():
            all_FLs.append(all_interpolated_FL[color][FL])

    # Initialize the best_FL_count dictionary
    best_FL_count = {}
    for r in range(0, 256, 5):
        for g in range(0, 256, 5):
            for b in range(0, 256, 5):
                best_FL_count[str((r, g, b))] = 0

    best_FL, _ = find_best_FL(all_interpolated_FL, dominant_color)
    best_FL_count[str(best_FL)] += 1

    print(best_FL[0], best_FL[1], best_FL[2])

    # Record the primaries under the best matching FL
    primaries_best_FL = []
    R_FL = []
    G_FL = []
    B_FL = []
    for i in range(0, 8, 1):
        primaries_best_FL.append(all_interpolated_FL[color_list[i]][best_FL])
        R_FL.append(all_interpolated_FL[color_list[i]][(255, 0, 0)])
        G_FL.append(all_interpolated_FL[color_list[i]][(0, 255, 0)])
        B_FL.append(all_interpolated_FL[color_list[i]][(0, 0, 255)])

    # Plot the image and FL gamut in 3D
    plot_image_and_FL_gamut(downsampled_pixels_lab, primaries_best_FL, dominant_color, selected_pixels_lab, all_FLs, G_FL, R_FL, B_FL)

    # Convert the primaries to Lab color space
    primaries_best_FL = np.array(primaries_best_FL)
    Lab_primaries_best_FL = []
    for color in primaries_best_FL:
        lab = colour.XYZ_to_Lab(color)
        Lab_primaries_best_FL.append(lab)

    # Outout the primaries under the best matching FL in txt file
    image_name = image_path.split('/')[-1].split('.')[0]
    with open(f'output/primaries/{image_name}_primaries.txt', 'w') as f:
        # print the imput image path
        f.write(f"Input image: {image_path}\n")
        # print the best matching FL
        f.write(f"Best FL: {best_FL}\n")
        # print the predicted primaries in Lab color space
        f.write(f"{Lab_primaries_best_FL[7][0]}\t{Lab_primaries_best_FL[7][1]}\t{Lab_primaries_best_FL[7][2]}\n")
        for i in range(0, 7, 1):
            f.write(f"{Lab_primaries_best_FL[i][0]}\t{Lab_primaries_best_FL[i][1]}\t{Lab_primaries_best_FL[i][2]}\n")


if __name__ == "__main__":
    # for i in range(1, 25, 1):
    #     kodim_num = str(i).zfill(2)
    #     image_path = f'input/image/kodim{kodim_num}.png'
    #     data_file_path = f'input/excel/i1_8colors_27FL_v1.xlsx'
    #     find_FL_with_mean(image_path, data_file_path)
	image_path = 'input/image/kodim01.png'
	data_file_path = 'input/excel/i1_8colors_27FL_v1.xlsx'
	# Set up logging
	logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
	find_FL_with_mean(image_path, data_file_path)
