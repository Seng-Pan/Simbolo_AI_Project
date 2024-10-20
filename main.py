import os
import json
import cv2
import numpy as np
from PIL import Image as Img
import pytesseract as pyt
import re
from dateutil import parser


pyt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  

image_dir = "C:/Users/Seng Pan/Can_Do_Crew_AI_Project/Payment_ImageToText/src/KBZ/"

# Regular expression patterns for extracting fields
desired_date_format = "%Y/%m/%d"

transtype_pattern = re.compile(r"(Transaction Type|Type)\s?:?\s?(.+)")
notes_pattern = re.compile(r"^(Notes|Note|Purpose|Reason|Remarks)\s?:?\s?(.+)")
transtime_pattern = re.compile(r"^(Transaction Time|Date and Time|Date & Time|Transaction Date|Bate and Time)\s?:?\s?(.+)")
transno_pattern = re.compile(r"^(Transaction No|Transaction ID|‘Transaction Code|Transaction IO)\s?:?\s?(.+)")
receiver_pattern = re.compile(r"(To|Receiver Name|Send To)\s?:?\s?([A-Za-z0-9\s]+)")
sender_pattern = re.compile(r"^(From|Sender Name|Send From)\s?:?\s?(.+)")
amount_data_pattern = re.compile(r"^(Amount|Total Amount|Amaunt)\s?:?\s?(.+)")


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

        # Threshold to convert the image to binary (black and white)
        # Adjust the threshold value to ensure better extraction of gray text
        _, thresh = cv2.threshold(sharpened, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Inpainting to remove the watermark
        result = cv2.inpaint(img, thresh, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Convert back to a PIL image
        pil_image = Img.fromarray(thresh)

        # Use Tesseract to do OCR on the image
        config = "--psm 6 --dpi 300"
        text = pyt.image_to_string(pil_image, config=config, lang='eng')
        return text
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
    
# Split text into lines
def split_text_into_lines(text):
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]

def extract_date_time(date_time_str):
    """
    Extracts date and time from the input string using regex and dateutil parser.

    :param date_time_str: String containing date and time
    :return: Tdate, time
    """

    # Define regular expressions to match different date and time formats
    date_pattern = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} \w+ \d{4}|\w+ \d{1,2}, \d{4})")
    time_pattern = re.compile(r"\b((1[0-2]|0?[1-9]):[0-5][0-9](?::[0-5][0-9])?\s?[APap][Mm]|(2[0-3]|[01]?[0-9]):[0-5][0-9](?::[0-5][0-9])?)\b")  

    try:
        # Search date and time matches in the input string    
        date_match = date_pattern.search(date_time_str)
        time_match = time_pattern.search(date_time_str)

        date_obj = parser.parse(date_match.group()) if date_match else None
        time_obj = parser.parse(time_match.group()) if time_match else None

        formatted_date = date_obj.strftime("%Y/%m/%d") if date_obj else ""
        formatted_time = time_obj.strftime("%H:%M:%S") if time_obj else ""
        
    except Exception as e:
           print(f"Error parsing date or time: {e}")

    return formatted_date, formatted_time

def extract_amount_only(amount_str):    
    """
    Extracts numeric amount from the amount string using regex.

    :param amount_str: amount with negative sign, MMK, Ks
    :return: numeric amount as a string
    """
    # formatted_amount = amount_str
    amount_only_pattern = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
    amount_match = amount_only_pattern.search(amount_str)
    
    return amount_match.group().replace("-", "").strip() if amount_match else None

def extract_transaction_data(text):
    
    transaction_data = {
        "Transaction No" : None,
        "Date": None,
        "Time": None,
        "Transaction Time" : None,
        "Transaction Type": None,
        "Sender Name": None,
        "Amount": None,
        "Receiver Name": None,
        "Notes": None
    }

    # Use regular expressions to find key information line by line
    lines = split_text_into_lines(text)
    for line in lines:
        # Transaction Time
        if re.search(transtime_pattern, line):
            transtime_pattern_match = transtime_pattern.search(line)
            date_time_str  = transtime_pattern_match.group(2).strip().strip('@').strip()
            transaction_data["Transaction Time"] = date_time_str            
            transaction_data["Date"], transaction_data["Time"] = extract_date_time(date_time_str)     

        # Transaction No
        elif re.search(transno_pattern, line):
             transno_pattern_match = transno_pattern.search(line)
             transaction_data["Transaction No"] = transno_pattern_match.group(2).strip().strip('@').strip()

        # Transaction Type
        elif re.search(transtype_pattern, line):
             transtype_pattern_match = transtype_pattern.search(line)
             transaction_type = transtype_pattern_match.group(2).strip().strip('@').strip()
             # Remove non-alphanumeric characters except spaces
             transaction_data["Transaction Type"] = re.sub(r"[^\w\s]", "", transaction_type)

        # Amounts
        elif re.search(amount_data_pattern, line):
             amount_data_pattern_match = amount_data_pattern.search(line)
             amount_string = amount_data_pattern_match.group(2).strip().strip('@').strip()
             transaction_data["Amount"] = extract_amount_only(amount_string)

        # Sender Name
        elif re.search(sender_pattern, line):
             sender_pattern_match = sender_pattern.search(line)
             transaction_data["Sender Name"] = sender_pattern_match.group(2).strip().strip('@').strip()

        # Receiver Name
        elif re.search(receiver_pattern, line):
             receiver_pattern_match = receiver_pattern.search(line)
             transaction_data["Receiver Name"] = receiver_pattern_match.group(2).strip().strip('@').strip()

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
            # Extract text using Tesseract
            extracted_text = extract_text_from_image(image_path)
            print(f"{extracted_text}")
            print(f"Extracted data from {filename}")

            # Extract transaction information using regex
            transaction_info = extract_transaction_data(extracted_text)
            print(transaction_info)

            # Add to list of all transactions
            all_transactions.append(transaction_info)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Save the extracted transaction data to a JSON file
output_json_path = "transactions_data.json"
with open(output_json_path, 'w') as json_file:
    json.dump(all_transactions, json_file, indent=4)

print(f"All data saved to {output_json_path}")


