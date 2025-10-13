from google import genai
from google.genai import types
from pathlib import Path
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

parts = load_parts("C:/Users/Evang/Desktop/Work/DS440/frames_output")

with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

client = genai.Client(api_key = api_key)
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=parts + [prompt]
  )

print(response.text)