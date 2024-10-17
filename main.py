import os
import json
import cv2
import numpy as np
from PIL import Image as Img
import pytesseract as pyt
import re


pyt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  


image_dir = "C:/Users/Seng Pan/Can_Do_Crew_AI_Project/Payment_ImageToText/src/KBZ/"

# Regular expression patterns for extracting fields
date_pattern = re.compile(
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{1,2} \w+ \d{4})|(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)
amount_pattern = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(MMK|Ks|Kyat)?")
send_from_pattern = re.compile(r"(From|Sender Name|Send From)\s?:?\s?([A-Za-z\s]+)")
send_to_pattern = re.compile(r"(To|Receiver Name|Send To)\s?:?\s?([A-Za-z\s]+)")
notes_pattern = re.compile(r"(Notes|Purpose)\s?:?\s?(.+)")
transaction_type_pattern = re.compile(r"\s?:?\s?Transfer")

def extract_text_from_image(image_path):
    """
    Extracts text from an image using Tesseract OCR.

    :param image_path: Path to the image file
    :return: Extracted text as a string, or None if extraction fails
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to reduce noise and smoothen the image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Increase contrast using adaptive histogram equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(blurred)

        # Sharpen the image to make text more readable
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced_img, -1, kernel)

        # Apply a threshold to convert the image to binary (black and white)
        # Adjust the threshold value to ensure better extraction of gray text
        _, thresh = cv2.threshold(sharpened, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to a PIL image
        pil_image = Img.fromarray(thresh)

        # Use Tesseract to do OCR on the image
        text = pyt.image_to_string(pil_image, lang='eng')
        return text
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def extract_transaction_data(text):
    
    transaction_data = {
        "Transaction Type": None,
        "Sender Name": None,
        "Amount": None,
        "Receiver Name": None,
        "Date": None,
        "Notes": None
    }

    # Use regular expressions to find key information
    # Transaction Type
    transaction_type_match = transaction_type_pattern.search(text)
    if transaction_type_match:
        transaction_data["Transaction Type"] = transaction_type_match.group(0).strip()

    # Sender Name
    sender_match = send_from_pattern.search(text)
    if sender_match:
        transaction_data["Sender Name"] = sender_match.group(2).strip()
    else:
        transaction_data["Sender Name"] = None  # Leave as null if not found

    # Amount
    amount_match = amount_pattern.search(text)
    if amount_match:
        transaction_data["Amount"] = amount_match.group(0).replace("—", "-").strip()

    # Receiver Name
    receiver_match = send_to_pattern.search(text)
    if receiver_match:
        transaction_data["Receiver Name"] = receiver_match.group(2).strip()

    # Date
    date_match = date_pattern.search(text)
    if date_match:
        transaction_data["Date"] = date_match.group(0).strip()

    # Notes
    notes_match = notes_pattern.search(text)
    if notes_match:
        transaction_data["Notes"] = notes_match.group(2).strip()

    return transaction_data


# Process images and save extracted data to JSON
all_transactions = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)

        try:
            # Image preprocessing using Pillow
            # image = Img.open(image_path)

            # Extract text using Tesseract
            extracted_text = extract_text_from_image(image_path)
            print(f"Extracted data from {filename}: \n{extracted_text}\n")

            # Extract transaction information using regex
            transaction_info = extract_transaction_data(extracted_text)
            transaction_info["File"] = filename  # Optional: Add filename for reference

            # Add to list of all transactions
            all_transactions.append(transaction_info)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Save the extracted transaction data to a JSON file
output_json_path = "transactions_data.json"
with open(output_json_path, 'w') as json_file:
    json.dump(all_transactions, json_file, indent=4)

print(f"All data saved to {output_json_path}")


