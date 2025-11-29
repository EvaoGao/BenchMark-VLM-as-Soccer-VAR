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


def extract_frames_center(video_url, output_dir="frames_output", center_duration=3.0, center_fps=20, outer_fps=10):
    """Extract frames from a video but sample the central portion more densely.

    - center_duration: length in seconds of the central window to sample at center_fps
    - center_fps: frames per second for the central window
    - outer_fps: frames per second for the rest of the video

    This function downloads the video like extract_frames_from_video and writes
    frames into output_dir. It returns the number of frames saved.
    """

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
        # cleanup temp file before raising
        try:
            os.unlink(temp_video.name)
        except Exception:
            pass
        raise RuntimeError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"fps: {fps:.2f}, frames: {total_frames}, duration: {duration:.2f} seconds")

    # determine central window
    if center_duration <= 0:
        center_duration = 0.0
    if center_duration >= duration:
        center_start = 0.0
        center_end = duration
    else:
        center_start = max(0.0, (duration - center_duration) / 2.0)
        center_end = center_start + center_duration

    frame_id = 0

    def _save_at_intervals(start_time, end_time, fps_sampling, start_id):
        nonlocal frame_id
        if end_time <= start_time:
            return
        interval = 1.0 / fps_sampling
        t = start_time
        # ensure we don't re-save an identical time due to floating rounding
        while t < end_time:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            success, frame = cap.read()
            if not success:
                break
            output_path = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_id += 1
            t += interval

    # sample before center
    _save_at_intervals(0.0, center_start, outer_fps, frame_id)
    # sample center more densely
    _save_at_intervals(center_start, center_end, center_fps, frame_id)
    # sample after center
    _save_at_intervals(center_end, duration, outer_fps, frame_id)

    cap.release()
    # cleanup temporary video file
    try:
        os.unlink(temp_video.name)
    except Exception:
        pass

    print(f"Saved {frame_id} images to: {output_dir}")
    return frame_id


if __name__ == "__main__":
    video_link = "https://varsfootball.s3.eu-west-3.amazonaws.com/Train/action_0/clip_1.mp4"  
    save_dir = "frames_output"
    frames_per_sec = 10
    # ============================

    extract_frames_center(video_link, save_dir)
