import pytesseract
import cv2
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = "text_image.jpeg"  # Path to the image file
image = cv2.imread(image_path)  # Read the image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale image

processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # Process the image

cv2.imwrite("processed_image.jpeg", processed_image)  # Save the processed image

text = pytesseract.image_to_string(processed_image)  # Extract text from the processed image

print("Extracted Text:\n", text)