from google import genai
from google.genai import types
from pathlib import Path
import frame_extraction
import os

valid_extensions = ('jpg', 'jpeg', 'png', 'gif', 'bmp')
api_key = "AIzaSyBA0Ba58PtBLKFMhTimoYfpRV4RzJi28ig"

MIME = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",
}

def get_extension(file_path):
    p = Path(file_path)
    return p.suffix.lstrip('.').lower()

def load_parts(folder_dir: str):
    parts_list = []
    for filename in os.listdir(folder_dir):
        extension = get_extension(filename)
        filepath = os.path.join(folder_dir, filename)

        if extension in valid_extensions:
            with open(filepath, 'rb') as f:
                image_bytes = f.read()
                parts_list.append(types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=MIME[extension],
                ))

    return parts_list

def gemini_call(URL1: str, URL2: str, prompt_path: str):
    frame_extraction.extract_frames_from_video(URL1,"frames_live")
    frame_extraction.extract_frames_from_video(URL2,"frames_replay")
    parts = load_parts("frames_live") + load_parts("frames_replay")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    client = genai.Client(api_key = api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=parts + [prompt]
    )

    return response.text

if __name__ == "__main__":
    response = gemini_call(
        "https://varsfootball.s3.eu-west-3.amazonaws.com/Test/action_3/clip_0.mp4",
        "https://varsfootball.s3.eu-west-3.amazonaws.com/Test/action_3/clip_1.mp4",
        "C:/Users/Evang/Desktop/Work/DS440/codes/prompt.txt"
    )
    print(response)