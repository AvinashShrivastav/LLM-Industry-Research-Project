# backend.py
import plotly.graph_objects as go
import random
import google.generativeai as genai
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import config

# Required Secret Key

# Get your key: https://ai.google.dev/
gemini_key = config.gemini_key

# Get your key: https://platform.openai.com/signup
open_ai_key = config.open_ai_key

# Input
pdf_loc = [r"2024.pdf"]


def process_image_to_json(image_path):
    # Simulate processing image to JSON
    gemini_img_output = get_image_info(image_path)
    jsondata = convert_table_to_json(gemini_img_output)
    return jsondata

def create_vector_db_from_pdf(pdf_path):
    # Simulate creating vector DB from PDF
    # Return True if vector DB is created successfully, otherwise False
    return True

def generate_updated_json(original_json):
    # Simulate generating updated JSON based on original JSON
    # Here, we simply add some random noise to the 'y' values
    updated_json = original_json.copy()
    # updated_json['y'] = [y + random.uniform(-1, 1) for y in original_json['y']]
    return updated_json

def plot_graph_from_json(jsondata, output_path):
    # Default metadata values
  title = "Graph"
  xaxis_title = "X-axis"
  yaxis_title = "Y-axis"
  legend_position = "top-right"
  legend_entries = []

  
  # Create data traces for the graph
  traces = []
  for data_point in jsondata.get("data", []):
    name = data_point.get("name")
    x = data_point.get("x")
    y = data_point.get("y")
    color = data_point.get("color")
    trace = go.Scatter(
      x=x,
      y=y,
      mode='lines+markers',
      name=name,
      marker=dict(color=color),
    )
    traces.append(trace)

  # Create the Plotly figure
  fig = go.Figure(data=traces)

  # Customize the figure
  fig.update_layout(
    title=title,
    xaxis_title=xaxis_title,
    yaxis_title=yaxis_title,
    legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1
    )
  )

  # Add legend entries if provided
  for entry in legend_entries:
    name = entry.get("name")
    color = entry.get("color")
    if name and color:
      fig.add_trace(
        go.Scatter(
          x=[None],
          y=[None],
          mode='markers',
          name=name,
          marker=dict(color=color),
          showlegend=True,
        )
      )
      
  # Return the figure
  fig.write_image(output_path) 

  

# Code to get image information
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

def get_image_info(image_loc):
    genai.configure(api_key=gemini_key)
    # Set up the model
    generation_config = {
        "temperature":0,
        "top_p":1,
        "top_k":32,
        "max_output_tokens":4096,
    }
    

    model = genai.GenerativeModel(model_name="gemini-pro-vision", generation_config=generation_config)

    input_prompt = """ You are an expert in reading the line charts. You will be given a line chart and you have to represent the line chart in tabular format along with all other meta data  """

    question_prompt = '''
        You are an expert in reading the line charts. You will be given a line chart and you have to represent the line chart in tabular format, adhering to the following guidelines:				
        1. The table's title will be derived from the chart's title. If no title is provided, it will be based on the labels of the x and y axes, or left as 'NA' if neither is available.				
        2. Identify the the labels on the x-axes and y-axes. If no label is provided name them according to your knowledge.				
        3. Identify the legends in the line chart and label them accordingly. If no legends are provided, they will be named as 'plot1', 'plot2', and so forth.				
        4. Only data points located directly on the x-axes will be considered.				
        5. Consider the steps size of the y-axes and estimate the value of the data point precisely.				
        6. For line charts with dual y-axes, determine whether the line plot corresponds to which y-axes, then extract the value of the data points accordingly.				
        7. Only add numerical values to the table, no units and characters. Example if the value of the data point is 10% then write only 10 in the table.
        8. If there is only one line consider it as a single plot and any different color point on the line as special marker
        '''

    image_prompt = input_image_setup(image_loc)
    prompt_parts = [input_prompt, image_prompt[0], question_prompt]
    response = model.generate_content(prompt_parts)
    return str(response.text)




prompt_table_to_json = '''Convert this table to JSON format: The json should clearly mention all the details of the table including the column names and the values in the table, the title, the x,y labels other legends and all the other details.
    one example of the format is, json must be enclosed in <json> tags. You may include/exclude the tags and number of attrubutes and datapoints  as per your requirement. The format of the json is as follows:
    <json>
    {
  "metadata": {
    "title": "title_here",
    "xaxis": {
      "title": "x_title_here"
    },
    "yaxis": {
      "title": "Y_title_here"
    }
  },
  "data": [
    {
      "name": "name of arrtibute",
      "x": [list of x values],
      "y": [list of y values if available]
    },
    {
      "name": "name of arrtibute",
      "x": [list of x values],
      "y": [list of y values if available]
    },
    {
      "name": "name of arrtibute",
      "x": [list of x values],
      "y": [list of y values if available]
    },
    {
      "name": "name of arrtibute",
      "x": [list of x values],
      "y": [list of y values if available]
  ]
}
<json>'''






def convert_table_to_json(table, prompt_table_to_json = prompt_table_to_json):
    # Configure the API client
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-pro')

    # Define the prompt
    prompt = prompt_table_to_json + table

    response = model.generate_content(prompt)
    # Print the generated text
    return str(response.text)
    
import plotly.graph_objects as go
import json

def plot_json_graph(json_data):
  # Default metadata values
  title = "Graph"
  xaxis_title = "X-axis"
  yaxis_title = "Y-axis"
  legend_position = "top-right"
  legend_entries = []

  
  # Create data traces for the graph
  traces = []
  for data_point in json_data.get("data", []):
    name = data_point.get("name")
    x = data_point.get("x")
    y = data_point.get("y")
    color = data_point.get("color")
    trace = go.Scatter(
      x=x,
      y=y,
      mode='lines+markers',
      name=name,
      marker=dict(color=color),
    )
    traces.append(trace)

  # Create the Plotly figure
  fig = go.Figure(data=traces)

  # Customize the figure
  fig.update_layout(
    title=title,
    xaxis_title=xaxis_title,
    yaxis_title=yaxis_title,
    legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1
    )
  )

  # Add legend entries if provided
  for entry in legend_entries:
    name = entry.get("name")
    color = entry.get("color")
    if name and color:
      fig.add_trace(
        go.Scatter(
          x=[None],
          y=[None],
          mode='markers',
          name=name,
          marker=dict(color=color),
          showlegend=True,
        )
      )

  # Return the figure
  return fig

import re
def extract_json_text(text):
  # Extract JSON text using regular expression
  match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)

  if match:
    json_text = match.group(1).strip()  # Extract the matched text and strip whitespace
    return json_text
  else:
    return "No JSON text found."

import plotly.graph_objects as go
import json


# # The third component is to read a variety of charts along with their metadata, generate tables 
# # and create input for the LLM to generate (this would require a vision model having the capability to read images with right prompting).
# gdp_img = "new graphs/linechart_other3.jpg"
# gemini_img_output = get_image_info(gdp_img)
# print("Gemini Output: Tabular : ")
# print(gemini_img_output)



# # The fifth component is to update the table and generate graph as final output 
# # (this would require usage of python's visualisation library to recreate the graphs).
# jsondata = convert_table_to_json(gemini_img_output)
# print("Gemini Output: json : ")
# print(jsondata)

# # Extract the JSON text (assuming your JSON data is within a string)
# json_text = extract_json_text(jsondata)
# parsed_json = json.loads(json_text) 

# # Generate the plot
# fig = plot_json_graph(parsed_json)
# fig.show()