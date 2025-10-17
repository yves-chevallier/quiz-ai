# pip install openai
import base64
from pathlib import Path
from openai import OpenAI

# --- Inputs (files you mentioned) ---
PROMPT_MD = Path("identify-question.md")  # contains the exact instructions/prompt
# IMAGE_FILE = Path("image.png")    # the scan to analyze
IMAGE_FILE = Path("../out/cut/2025-10-16-14-46-25_page4_1.jpg")  # the scan to analyze

# --- Load prompt and image ---
prompt_text = PROMPT_MD.read_text(encoding="utf-8")

image_b64 = base64.b64encode(IMAGE_FILE.read_bytes()).decode("utf-8")
image_data_uri = f"data:image/png;base64,{image_b64}"

# --- Create client ---
client = OpenAI()

# --- Build and send the exact request ---
# Model: same as this assistant
MODEL_NAME = "gpt-5"

resp = client.responses.create(
    model=MODEL_NAME,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": image_data_uri},
            ],
        }
    ],
)

# --- Use the text output (the JSON you requested from the prompt) ---
output_text = resp.output_text

# Print to console
print(output_text)

# Optionally persist the result
Path("analysis_result.json").write_text(output_text, encoding="utf-8")
print("\nSaved to analysis_result.json")
