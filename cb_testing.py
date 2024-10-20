import os
import json
import cv2
import numpy as np
from PIL import Image as Img
import pytesseract as pyt
import re
from dateutil import parser

pyt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  

image_dir = "C:/Users/Seng Pan/Can_Do_Crew_AI_Project/Payment_ImageToText/src/CB/"

# Regular expression patterns for extracting fields
transtype_pattern = re.compile(r"(Transfer Complete!|Transaction Type|Type)\s?:?\s?(.+)")
notes_pattern = re.compile(r"(Reason|Notes|Purpose)\s?:?\s?(.+)")
transtime_pattern = re.compile(r"(Transaction Date|Date and Time|Transaction Time)\s?:?\s?(.+)")
transno_pattern = re.compile(r"(Transaction ID|Transaction No)\s?:?\s?(.+)")
receiver_pattern = re.compile(r"(Transfer to|Receiver|To)\s?:?\s?([A-Za-z0-9\s]+)")
sender_pattern = re.compile(r"(Transfer from|Sender|From)\s?:?\s?(.+)")
amount_data_pattern = re.compile(r"(Amount)\s?:?\s?([0-9,.]+)")

# Function to extract text from image
def extract_text_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(blurred)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced_img, -1, kernel)

        # Apply inpainting to remove watermark interference
        _, thresh = cv2.threshold(sharpened, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil_image = Img.fromarray(thresh)

        config = "--psm 6"
        text = pyt.image_to_string(pil_image, config=config, lang='eng')
        return text

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to split text into lines
def split_text_into_lines(text):
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]

# Extract and format date and time
def extract_date_time(date_time_str):
    date_pattern = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} \w+ \d{4}|\w+ \d{1,2}, \d{4})")
    time_pattern = re.compile(r"\b((1[0-2]|0?[1-9]):[0-5][0-9](?::[0-5][0-9])?\s?[APap][Mm]|(2[0-3]|[01]?[0-9]):[0-5][0-9](?::[0-5][0-9])?)\b")

    try:
        date_match = date_pattern.search(date_time_str)
        time_match = time_pattern.search(date_time_str)

        date_obj = parser.parse(date_match.group()) if date_match else None
        time_obj = parser.parse(time_match.group()) if time_match else None

        formatted_date = date_obj.strftime("%Y/%m/%d") if date_obj else ""
        formatted_time = time_obj.strftime("%H:%M:%S") if time_obj else ""

    except Exception as e:
        print(f"Error parsing date or time: {e}")
        return None, None

    return formatted_date, formatted_time

# Extract amount (numeric value)
def extract_amount_only(amount_str):
    amount_only_pattern = re.compile(r"-?\d*(?:,\d*)*(?:\.\d{2})?")
    amount_match = amount_only_pattern.search(amount_str)
    return amount_match.group().replace("-", "").strip() if amount_match else None

# Main function to extract transaction data
def extract_transaction_data(text):
    transaction_data = {
        "Transaction No": None,
        "Date": None,
        "Time": None,
        "Transaction Type": None,
        "Sender Name": None,
        "Amount": None,
        "Receiver Name": None,
        "Notes": None
    }

    lines = split_text_into_lines(text)
    for line in lines:
        # Transaction Time and Date
        if re.search(transtime_pattern, line):
            transtime_match = transtime_pattern.search(line)
            date_time_str = transtime_match.group(2).strip()
            transaction_data["Date"], transaction_data["Time"] = extract_date_time(date_time_str)

        # Transaction ID
        elif re.search(transno_pattern, line):
            transno_match = transno_pattern.search(line)
            transaction_data["Transaction No"] = transno_match.group(2).strip()

        # Transaction Type
        elif re.search(transtype_pattern, line):
            transtype_match = transtype_pattern.search(line)
            transaction_data["Transaction Type"] = transtype_match.group(2).strip()

        # Amount
        elif re.search(amount_data_pattern, line):
            amount_match = amount_data_pattern.search(line)
            transaction_data["Amount"] = extract_amount_only(amount_match.group(2))

        # Sender Name
        elif re.search(sender_pattern, line):
            sender_match = sender_pattern.search(line)
            transaction_data["Sender Name"] = sender_match.group(2).strip()

        # Receiver Name
        elif re.search(receiver_pattern, line):
            receiver_match = receiver_pattern.search(line)
            transaction_data["Receiver Name"] = receiver_match.group(2).strip()

        # Notes
        elif re.search(notes_pattern, line):
            notes_match = notes_pattern.search(line)
            transaction_data["Notes"] = notes_match.group(2).strip()

    return transaction_data

# Process images and save extracted data to JSON
all_transactions = []
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)

        try:
            extracted_text = extract_text_from_image(image_path)
            transaction_info = extract_transaction_data(extracted_text)
            print(transaction_info)
            all_transactions.append(transaction_info)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Save the extracted transaction data to a JSON file
output_json_path = "transactions_data.json"
with open(output_json_path, 'w') as json_file:
    json.dump(all_transactions, json_file, indent=4)

print(f"All data saved to {output_json_path}")
