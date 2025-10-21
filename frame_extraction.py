import cv2
import os
import math
import requests
import tempfile
import shutil

def extract_frames_from_video(video_url, output_dir="frames_output", frames_per_second=20):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with requests.get(video_url, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            temp_video.write(chunk)
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"fps: {fps:.2f}, frames: {total_frames}, duration: {duration:.2f} seconds")

    frame_interval = 1 / frames_per_second 
    current_time = 0.0
    frame_id = 0

    while current_time < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        success, frame = cap.read()
        if not success:
            break

        output_path = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(output_path, frame)
        frame_id += 1

        current_time += frame_interval

    cap.release()
    print(f"Saved {frame_id} images to: {output_dir}")


if __name__ == "__main__":
    video_link = "https://varsfootball.s3.eu-west-3.amazonaws.com/Train/action_0/clip_1.mp4"  
    save_dir = "frames_output"
    frames_per_sec = 10
    # ============================

    extract_frames_from_video(video_link, save_dir, frames_per_sec)
