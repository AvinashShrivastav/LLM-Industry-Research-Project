# Code to get image information
import google.generativeai as genai
import config
def input_image_setup(file_loc):
    from pathlib import Path

    if not (img := Path(file_loc)).exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": Path(file_loc).read_bytes()
            }
        ]
    return image_parts
def get_image_info(image_loc, prompt):
    genai.configure(api_key=config.gemini_key)
    # Set up the model
    generation_config = {
        "temperature":0,
        "top_p":1,
        "top_k":32,
        "max_output_tokens":4096,
    }
    
    model = genai.GenerativeModel(model_name="gemini-pro-vision", generation_config=generation_config)

    input_prompt = """ You are an expert in data visualization and graph analysis, adept at interpreting graphical data and generating structured JSON configurations for Plotly"""

    question_prompt = prompt

    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt, image_prompt[0], question_prompt]
    response = model.generate_content(prompt_parts)
    return str(response.text)
