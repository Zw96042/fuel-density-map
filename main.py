import os
import analysis
from PIL import Image

def raw_data_analysis():
    # Example usage
    vid_path = "videos/citrus_bread_wr.mp4"
    output_dir = "output"

    start_time = 5  # in seconds
    end_time = 171   # in seconds
    
    print("Analyzing video...")
    vid_totals = analysis.get_total_yellow_pixels_from_video(vid_path, start_time, end_time)
    # Save the raw results to a file

    print("Saving results...")
    raw_data_file_name = vid_path.split("/")[-1].split(".")[0] + "_raw_data.txt"
    with open(os.path.join(output_dir, raw_data_file_name), "w") as f:
        for row in vid_totals:
            f.write(" ".join(str(x) for x in row) + "\n")
    print("Results saved.")

def process_raw_data():
    data_path = "output/citrus_bread_wr_raw_data.txt"
    processed_image_path = "images"
    average_display_color = (255, 0, 255) # magenta, easy to see

    print("Processing raw data...")
    # Read the raw data from the file
    with open(data_path, "r") as f:
        raw_data = [list(map(int, line.split())) for line in f]
    
    processed_folder_name = data_path.split("/")[-1].split("_")[0] + "_processed"
    processed_folder_path = os.path.join(processed_image_path, processed_folder_name)
    os.makedirs(processed_folder_path, exist_ok=True)

    max_value = max(max(row) for row in raw_data)
    average_of_non_zero_values = sum(x for row in raw_data for x in row if x > 0) / sum(1 for row in raw_data for x in row if x > 0)
    print(f"Max value: {max_value}")
    print(f"Average of non-zero values: {average_of_non_zero_values}")

    # turn the raw data into an array of colors where the color is determined by the value of the raw data at that point, 
    # with 0 being black, 
    # max_value being white,
    #  average_of_non_zero_values being average_display_color
    # and values between 0 and average_of_non_zero_values being a gradient from black to average_display_color,
    # and values between average_of_non_zero_values and max_value being a gradient from average_display_color to white

    color_array = []
    for row in raw_data:
        color_row = [get_color(value, max_value, average_of_non_zero_values, average_display_color) for value in row]
        color_array.append(color_row)

    print("Creating image...")
    # Save the color array as an image
    img = Image.new("RGB", (len(color_array[0]), len(color_array)))
    for i in range(len(color_array)):
        for j in range(len(color_array[i])):
            img.putpixel((j, i), color_array[i][j])
    img.save(os.path.join(processed_folder_path, "processed_image.png"))
    print("Image saved.")

    print("Creating transparent image...")
    # also create a version where black is transparent and and everything else is half transparent
    img = Image.new("RGBA", (len(color_array[0]), len(color_array)))
    for i in range(len(color_array)):
        for j in range(len(color_array[i])):
            if color_array[i][j] == (0, 0, 0):
                img.putpixel((j, i), (0, 0, 0, 0)) # transparent
            else:
                img.putpixel((j, i), color_array[i][j] + (128,)) # half transparent
    img.save(os.path.join(processed_folder_path, "processed_image_transparent.png"))
    print("Transparent image saved.")

def get_color(value, max_value, average_of_non_zero_values, average_display_color):
        if value == 0:
            return (0, 0, 0) # black
        elif value >= max_value:
            return (255, 255, 255) # white
        elif value <= average_of_non_zero_values:
            # gradient from black to average_display_color
            ratio = value / average_of_non_zero_values
            return tuple(int(ratio * average_display_color[i]) for i in range(3))
        else:
            # gradient from average_display_color to white
            ratio = (value - average_of_non_zero_values) / (max_value - average_of_non_zero_values)
            return tuple(int(average_display_color[i] + ratio * (255 - average_display_color[i])) for i in range(3))
        


if __name__ == "__main__":
    # raw_data_analysis()
    process_raw_data()