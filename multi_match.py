
import numpy as np
import analysis
import main

def get_multi_match_raw_data(video_path_array, start_time_array=None, end_time_array=None):
    if start_time_array is None or len(start_time_array) != len(video_path_array):
        start_time_array = [0] * len(video_path_array)
    if end_time_array is None or len(end_time_array) != len(video_path_array):
        end_time_array = [None] * len(video_path_array)

    all_frame_counts = []
    for video_path, start_time, end_time in zip(video_path_array, start_time_array, end_time_array):
        frame_counts = analysis.analyze_video_progression(video_path, start_time, end_time)
        all_frame_counts.append(frame_counts)

    return all_frame_counts

def get_multi_match_combined_data(video_path_array, start_time_array=None, end_time_array=None):
    all_frame_counts = get_multi_match_raw_data(video_path_array, start_time_array, end_time_array)
    combined_data = np.sum(all_frame_counts, axis=0)
    return combined_data

def get_multi_match_color_data(video_path_array, start_time_array=None, end_time_array=None, average_display_color=(255, 255, 0)):
    combined_data = get_multi_match_combined_data(video_path_array, start_time_array, end_time_array)
    max_value = np.max(combined_data)
    average_of_non_zero_values = np.mean(combined_data[combined_data > 0]) if np.any(combined_data > 0) else 0
    color_data = main.create_color_array(combined_data, max_value, average_of_non_zero_values, average_display_color)
    return color_data