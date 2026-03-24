import os

import cv2
import numpy as np
import analysis
from PIL import Image

vid_path = "videos/citrus_bread_wr_blackboxed.mp4"
start_time = 0  # in seconds
end_time = None   # in seconds
pct_from_average_to_max = 0.5 # how much of the way from the average to the max value should the average color be, as a percentage
average_display_color = (255, 0, 255) # magenta, easy to see

def raw_data_analysis():
    # Example usage
    output_dir = "output"

    
    
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
    data_path = vid_path.split("/")[-1].split(".")[0] + "_raw_data.txt"
    processed_image_path = "images"

    print("Processing raw data...")
    # Read the raw data from the file
    with open(data_path, "r") as f:
        raw_data = [list(map(int, line.split())) for line in f]
    
    processed_folder_name = vid_path.split("/")[-1].split(".")[0] + "_processed"
    processed_folder_path = os.path.join(processed_image_path, processed_folder_name)
    os.makedirs(processed_folder_path, exist_ok=True)

    max_value = max(max(row) for row in raw_data)
    actual_average = sum(x for row in raw_data for x in row if x > 0) / sum(1 for row in raw_data for x in row if x > 0)
    average_of_non_zero_values = actual_average + pct_from_average_to_max * (max_value - actual_average)
    print(f"Max value: {max_value}")
    print(f"Actual average of non-zero values: {actual_average}")
    print(f"Used Average of non-zero values: {average_of_non_zero_values}")

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
        return (0, 0, 0)  # black

    if max_value == 0:
        return average_display_color

    if value >= max_value:
        return (255, 255, 255)  # white

    # Avoid division by zero
    if average_of_non_zero_values == 0:
        return average_display_color

    if value <= average_of_non_zero_values:
        # gradient from black to average_display_color
        ratio = value / average_of_non_zero_values
        return tuple(int(ratio * average_display_color[i]) for i in range(3))
    else:
        # gradient from average_display_color to white
        denom = (max_value - average_of_non_zero_values)
        if denom == 0:
            return average_display_color

        ratio = (value - average_of_non_zero_values) / denom
        return tuple(
            int(average_display_color[i] + ratio * (255 - average_display_color[i]))
            for i in range(3)
        )       

 
def process_into_video_progression():
    frame_counts = analysis.analyze_video_progression(vid_path, start_time, end_time)
    output_dir = "" #"videos/progression/"
    output_name = output_dir + vid_path.split("/")[-1].split(".")[0] + "_processed_video.mp4"

    print("Processing video progression...")
    # convert the frame counts into color arrays
    frame_color_arrays = []
    for frame_count in frame_counts:
        print(f"Processing frame {len(frame_color_arrays)}/{len(frame_counts)}", end="\r")
        max_value = max(max(row) for row in frame_count)
        average_of_non_zero_values = sum(x for row in frame_count for x in row if x > 0) / sum(1 for row in frame_count for x in row if x > 0)
        average_of_non_zero_values = average_of_non_zero_values + pct_from_average_to_max * (max_value - average_of_non_zero_values)
        color_array = []
        for row in frame_count:
            color_row = [get_color(value, max_value, average_of_non_zero_values, average_display_color) for value in row]
            color_array.append(color_row)
        frame_color_arrays.append(color_array)
    
    # turn the color arrays and turn them into a video
    video_array = np.array(frame_color_arrays)
    array_to_video(video_array, output_name, fps=30)

    # # get the frames now with the black parts transparent and everything else half transparent
    # transparent_frame_color_arrays = []
    # for color_array in frame_color_arrays:
    #     transparent_color_array = []
    #     for row in color_array:
    #         transparent_color_row = []
    #         for color in row:
    #             if color == (0, 0, 0):
    #                 transparent_color_row.append((0, 0, 0, 0)) # transparent
    #             else:
    #                 transparent_color_row.append(color + (128,)) # half transparent
    #         transparent_color_array.append(transparent_color_row)
    #     transparent_frame_color_arrays.append(transparent_color_array)

    # transparent_video_array = np.array(transparent_frame_color_arrays)
    # array_to_video(transparent_video_array, output_name.replace(".mp4", "_transparent.mp4"), fps=30)

def array_to_video(video_array, output_path, fps=30):
    print("VIDEO ARRAY SHAPE:", video_array.shape)

    num_frames, height, width, channels = video_array.shape
    
    if num_frames == 0:
        raise ValueError("video_array has 0 frames — nothing to write")

    assert channels == 3, "Must have 3 color channels (RGB)"
    
    if video_array.dtype != np.uint8:
        video_array = video_array.astype(np.uint8)

    print("Pixel range:", video_array.min(), video_array.max())

    # Try H.264 first
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("avc1 failed, falling back to XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace(".mp4", ".avi")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError("All codecs failed")

    for i in range(num_frames):
        frame_rgb = video_array[i]

        # Sanity check
        if frame_rgb.shape != (height, width, 3):
            raise ValueError(f"Frame shape mismatch: {frame_rgb.shape}")

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print("Saved to:", output_path)

if __name__ == "__main__":
    # raw_data_analysis()
    # process_raw_data()
    process_into_video_progression()