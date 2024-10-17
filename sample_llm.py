import os
from PIL import Image as Img
import pytesseract as pyt
import json
from transformers import pipeline     # Hugging Face

pyt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe' 

ner_model = pipeline('ner', model="dslim/bert-base-NER")

image_dir = "C:/Users/Seng Pan/Can_Do_Crew_AI_Project/Payment_ImageToText/src/assets/"

# Extract and format text data
def extract_and_format_text(image_path): 

    try:
        # Image preprocessing using Pillow
        image = Img.open(image_path)

        # Extract data using Tesseract
        extracted_text = pyt.image_to_string(image)
        print(f"Extracted data from {os.path.basename(image_path)}: \n{extracted_text}\n")

        # Get entities using Hugging Face ner model
        entities = ner_model(extracted_text)

        # Structure format
        result_dict = {}
        for entity in entities:
            entity_type = entity['entity'].replace('B-', '').replace('I-', '')
            if entity_type not in result_dict:
                result_dict[entity_type] = entity['word']
            else:
                result_dict[entity_type] += ' ' + entity['word']

        return result_dict
    
    except Exception as e:
        print(f"Failed to process {os.path.basename(image_path)}: {e}")
        return {}

all_data = []
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png','.jpg','.jpeg')):
        image_path = os.path.join(image_dir, filename)

        formatted_data = extract_and_format_text(image_path)
        all_data.append(formatted_data)

# Save all data to a JSON file
output_json_path = "payment_data.json"
with open(output_json_path, 'w') as json_file:
    json.dump(all_data, json_file, indent = 4)
    

print(f"All data saved to {output_json_path}")