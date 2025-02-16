# Dependencies: pytesseract, cv2, PIL, langdetect, spellchecker
import pytesseract
import cv2
from PIL import Image
from langdetect import detect, DetectorFactory, detect_langs
from spellchecker import SpellChecker
from googletrans import Translator
import numpy as np

# Path to the image file
image_path = "test_image.png"

# Read the image  
image = cv2.imread(image_path)  

# Preprocess image for better OCR
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# Apply Gaussian Blur (reduces noise)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive Thresholding
processed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply Morphological Transformations to remove noise
kernel = np.ones((1, 1), np.uint8)
processed_image = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

# Save the processed image
cv2.imwrite("processed_image.png", processed_image)  

# Raw image
unprocessed_image = Image.open(image_path)

# Extract text from processed image with Latin bias
custom_config = "--psm 3 -l eng"
text = pytesseract.image_to_string(processed_image, config=custom_config)  

# Detect language
DetectorFactory.seed = 0
detected_language = detect(text)

# Print the detected language
print("Detected Language:", detected_language)
print("Detected Language Probabilities:", detect_langs(text))
# Split into lines
lines = text.splitlines()

# Initialize spell checker``
spell = SpellChecker()

#Load latin dictionary
#spell.word_frequency.load_text_file("vocabulary.txt")

# Correct the lines
correct_lines = [
    " ".join([spell.correction(word) if spell.correction(word) else word for word in line.split()])
    for line in text.splitlines()
]
correct_lines_string = "\n".join(correct_lines)

# Translate the original text
# translated_original = Translator().translate(text, src = "la", dest="en")
# translated_corrected = Translator().translate(correct_lines_string, src = "la", dest="en")

print("Original Text:\n", text)
print("\nCorrected Text:\n", "\n".join(correct_lines))

# print("\nTranslated Original:\n", translated_original)
# print("\nTranslated Corrected:\n", translated_corrected)
