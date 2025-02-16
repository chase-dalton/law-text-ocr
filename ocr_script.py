# Dependencies: pytesseract, cv2, PIL, langdetect, spellchecker
import pytesseract
import cv2
from PIL import Image
from langdetect import detect, DetectorFactory, detect_langs
from spellchecker import SpellChecker
import re

# Path to the image file
image_path = "law_text.png"

# Read the image  
image = cv2.imread(image_path)  

# Preprocess image for better OCR
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save the processed image
cv2.imwrite("law_text_processed.png", gray)  

# Raw image
unprocessed_image = Image.open(image_path)

# Extract text from processed image with Latin bias
custom_config = "--psm 3 -l eng+lat"
text = pytesseract.image_to_string(gray, config=custom_config)  

# Detect language
DetectorFactory.seed = 0
detected_language = detect(text)

# Print the detected language
print("Detected Language:", detected_language)
print("Detected Language Probabilities:", detect_langs(text))
# Split into lines
lines = text.splitlines()

# Initialize spell checker
spell = SpellChecker()


# Correct the lines
correct_lines = [
    " ".join([spell.correction(word) if spell.correction(word) else word for word in line.split()])
    for line in text.splitlines()
]
correct_lines_string = "\n".join(correct_lines)

print("Original Text:\n", text)
print("\nCorrected Text:\n", "\n".join(correct_lines))

