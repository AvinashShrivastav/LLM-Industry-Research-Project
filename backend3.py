# backend.py
import config
from plot_from_json_static import plot_from_ai_output
from image_to_json import get_image_info
from create_vector_db import convert_pdf_to_vector_db
from update_json import gpt_model_with_knowledge, gemini_model_with_knowledge


#STEP 1: CONVERT IMAGE TO JSON
def process_image_to_json(image_path,prompt):
    # Simulate processing image to JSON
    jsondata= get_image_info(image_path, prompt)
    return jsondata

#STEP 2: MAKE SUITABLE VECTOR DB
def create_vector_db_from_pdf(pdf_path):
    # Simulate creating vector DB from PDF
    vector_store = convert_pdf_to_vector_db(pdf_path)
    return vector_store

#STEP 3: UPDATE JSON FROM VECTOR DB
def generate_updated_json_gpt(original_json,prompt, vector_store):
    # Simulate generating updated JSON based on original JSON
    updated_json_gpt = gpt_model_with_knowledge(vector_store, original_json, prompt)
    return updated_json_gpt

def generate_updated_json_gemini(original_json,prompt, vector_store):
    # Simulate generating updated JSON based on original JSON
    updated_json_gemini = gemini_model_with_knowledge(vector_store, original_json, prompt)
    return updated_json_gemini

#STEP 4: PLOT UPDATED JSON
def plot_graph_from_json(jsondata, output_path):
    plot_from_ai_output(jsondata, output_path)
    return True


