import json
import logging
import os
import re
import numpy as np

import cv2
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Regular expression patterns for extracting fields
date_pattern = re.compile(
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{1,2} \w+ \d{4})|(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)
number_pattern = re.compile(
    r"(Transaction(?:\sID|\.? No\.?)|Reference ID)\s?:?\s?([\w\d]+)"
)
# Updated amount regex to handle negative values and special characters
amount_pattern = re.compile(r"[—-]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s?(MMK|Ks|Kyat)?")
send_from_pattern = re.compile(r"(From|Sender Name|Send From)\s?:?\s?([A-Za-z\s]+)")
send_to_pattern = re.compile(r"(To|Receiver Name|Send To)\s?:?\s?([A-Za-z0-9\s]+)")
notes_pattern = re.compile(r"(Notes|Purpose)\s?:?\s?(.+)")


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
        pil_image = Image.fromarray(thresh)

        # Use Tesseract to do OCR on the image
        text = pytesseract.image_to_string(pil_image, lang='eng')
        return text

        return text
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None


def extract_transaction_details(text):
    """
    Extract key fields (transaction date, number, amount, sender, receiver, notes) from the given text.

    :param text: Text extracted from an image
    :return: Dictionary of extracted transaction details
    """
    details = {
        "transaction_date": None,
        "transaction_number": None,
        "amount": None,
        "send_from": None,
        "send_to": None,
        "notes": None,
    }

    # Find and store each field using regex
    date_match: re.Match[str] | None = date_pattern.search(text)
    if date_match:
        details["transaction_date"] = date_match.group(0)

    number_match = number_pattern.search(text)
    if number_match:
        details["transaction_number"] = number_match.group(2)

    # Improve the amount extraction, handling special characters and negative values
    amount_match = amount_pattern.search(text)
    if amount_match:
        # Clean up special characters like "—"
        details["amount"] = amount_match.group(0).replace("—", "-").strip()

    send_from_match = send_from_pattern.search(text)
    if send_from_match:
        details["send_from"] = send_from_match.group(2).strip()

    send_to_match = send_to_pattern.search(text)
    if send_to_match:
        details["send_to"] = send_to_match.group(2).strip()

    notes_match = notes_pattern.search(text)
    if notes_match:
        details["notes"] = notes_match.group(2).strip()

    return details


def is_image_file(filename):
    """
    Checks if a given file is an image based on the extension.

    :param filename: Name of the file
    :return: True if the file is an image, False otherwise
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    return filename.lower().endswith(valid_extensions)


def main():
    # Directory containing the images
    image_dir = "C:/Users/Seng Pan/Can_Do_Crew_AI_Project/Payment_ImageToText/src/KBZ/"

    # Initialize a list to hold all transaction details
    all_transaction_details = []

    # Process all images in the directory
    for filename in os.listdir(image_dir):
        if is_image_file(filename):
            image_path = os.path.join(image_dir, filename)

            # Extract text from the image
            extracted_text = extract_text_from_image(image_path)

            if extracted_text:
                logging.info(f"Successfully processed {filename}")
                logging.info(f"Extracted text: {extracted_text}")
                logging.info("-" * 50)

                # Extract transaction details using regex
                transaction_details = extract_transaction_details(extracted_text)
                transaction_details["image_file"] = (
                    filename  # Add filename for reference
                )

                # Append the extracted details to the list
                all_transaction_details.append(transaction_details)
            else:
                logging.warning(f"Failed to extract text from {filename}")

    # Save all transaction details to a JSON file
    if all_transaction_details:
        output_json_file = "transaction_details.json"
        with open(output_json_file, "w") as json_file:
            json.dump(all_transaction_details, json_file, indent=4)
        logging.info(f"All transaction details saved to {output_json_file}")
    else:
        logging.warning("No transaction details were extracted from any images.")


if __name__ == "__main__":
    main()