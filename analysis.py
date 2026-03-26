import cv2
import numpy as np


def analyze_video_progression(vid_path, start_time=0, end_time=None):
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if end_time is None:
        end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_frames = end_frame - start_frame

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_counts = np.zeros(
        (total_frames, frame_height, frame_width),
        dtype=np.uint16
    )

    running_total = np.zeros((frame_height, frame_width), dtype=np.uint16)

    frame_index = 0

    while frame_index < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        mask = cv2.inRange(frame, (0,200,200), (100,255,255))
        running_total += mask // 255

        frame_counts[frame_index] = running_total
        frame_index += 1

    cap.release()
    return frame_counts