import logging
import os

import cv2
import openpyxl
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_text_from_image(image_path):
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use Tesseract to do OCR on the image
        text = pytesseract.image_to_string(Image.fromarray(gray))

        return text
    
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None


def save_to_excel(data, output_file):
    try:
        # Create a new workbook and select the active sheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active

        # Write the data to the sheet
        for row, line in enumerate(data.split("\n"), start=1):
            sheet.cell(row=row, column=1, value=line)

        # Save the workbook
        workbook.save(output_file)
    except Exception as e:
        logging.error(f"Error saving to Excel file {output_file}: {str(e)}")


def is_image_file(filename):
    # List of accepted image extensions (case-insensitive)
    valid_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    return filename.lower().endswith(valid_extensions)


def main():
    # Directory containing the images
    image_dir = "C:/Users/Seng Pan/Can_Do_Crew_AI_Project/Payment_ImageToText/src/assets/"

    # Output Excel file
    output_file = "output.xlsx"
    all_extracted_text = ""

    # Process all images in the directory
    for filename in os.listdir(image_dir):
        if is_image_file(filename):
            image_path = os.path.join(image_dir, filename)

            # Extract text from the image
            extracted_text = extract_text_from_image(image_path)

            if extracted_text:
                all_extracted_text += f"--- {filename} ---\n{extracted_text}\n\n"
                logging.info(f"Successfully processed {filename}")
            else:
                logging.warning(f"Failed to extract text from {filename}")

    # Save all extracted text to Excel
    if all_extracted_text:
        save_to_excel(all_extracted_text, output_file)
        logging.info(f"All processed text saved to {output_file}")
    else:
        logging.warning("No text was extracted from any images.")


if __name__ == "__main__":
    main()