import os
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from dataclasses import dataclass
from typing import List, Tuple
from PDFSNIPPER import split_pdf
from PDFSNIPPER import save_pages_as_images
import shutil

# -------------------- Only for Windows If Poppler not on PATH --------------------
#                      Set the path to the poppler binaries
os.environ['PATH'] += os.pathsep + r'C:\Program Files\Release-24.08.0-0\poppler-24.08.0\Library\bin'
# -------------------- Singleton OCR Class --------------------
class OCRSingleton:
    """Ensures that the PaddleOCR model is initialized only once."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Initializing OCR Model...")
            cls._instance = super(OCRSingleton, cls).__new__(cls)
            cls._instance.ocr = PaddleOCR(use_angle_cls=True, lang="en")
        return cls._instance

# -------------------- Dataclass for OCR Results --------------------
@dataclass
class OCRResult:
    """Stores the text detected and its bounding box coordinates."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

# -------------------- Dataclass for Temporary File Paths --------------------
@dataclass
class DEFAULT:
  """
    Stores the paths of the intermediate folders that are created while
    running the script.
  """
  single_page_pdf: str = "single_page_pdf"
  input_reports: str = "input_reports"
  output_reports: str = "output_reports"
  input: str = "input"
# -------------------- Utility Class --------------------
class FileHandler:
    """Handles file operations like reading images and saving results."""

    @staticmethod
    def get_images_from_folder(input_folder: str) -> List[str]:

        """
          Retrieve all image file paths from the input folder.

          Args:
                input_folder: The folder containing images of the pdf after calling the save_pages_as_images function
          Returns:
                A list of image file paths.
        """
        valid_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        return [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(valid_extensions)
        ]

    @staticmethod
    def delete_folder_securely(folder_path: str):
        """Securely deletes all files and subfolders within a folder."""
        try:
            if not os.path.exists(folder_path):
                print(f"Folder '{folder_path}' does not exist.")
                return

            for root, dirs, files in os.walk(folder_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")

                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Deleted subfolder: {dir_path}")
                    except Exception as e:
                        print(f"Error deleting subfolder {dir_path}: {e}")

            os.rmdir(folder_path)
            print(f"Deleted folder: {folder_path}")
        except Exception as e:
            print(f"Unexpected error while deleting folder: {e}")

    @staticmethod
    def save_to_csv(data: List[List[str]], output_folder: str, filename: str):
        """
          Save extracted text to a CSV file.

          Args:
              data: List of lists containing the structured text.
              output_folder: Folder where the CSV file will be saved (would be the input file name).
              filename: Name of the CSV file (would be the single page of theinput file that has been processed).
        """
        os.makedirs(output_folder, exist_ok=True)
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_folder, f"{filename}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {csv_path}")

# -------------------- OCR Processing Class --------------------
class OCRProcessor:
    """Handles OCR text extraction."""

    def __init__(self):
        """Retrieve the singleton OCR model instance."""
        self.ocr = OCRSingleton().ocr

    def extract_text_from_image(self, image) -> List[OCRResult]:
        """Perform OCR on an image and return structured OCRResult objects."""
        ocr_results = self.ocr.ocr(image, cls=True)
        if not ocr_results:
            print("No text detected!")
            return []

        extracted_data = []
        for result in ocr_results:
            for line in result:
                bbox, (text, prob) = line
                x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
                x_max, y_max = int(bbox[2][0]), int(bbox[2][1])
                extracted_data.append(OCRResult(text=text, bbox=(x_min, y_min, x_max, y_max)))

        return extracted_data

    def visualize_bounding_boxes(self, image, rows):
        """Visualize the bounding boxes on the image."""
        for row in rows:
          for box in row:
            x_min, y_min, x_max, y_max, text = box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green for structured boxes
            cv2.putText(image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# -------------------- Image Preprocessing Class --------------------
class ImagePreprocessor:
    """Handles image preprocessing for better OCR performance."""

    @staticmethod
    def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess image (grayscale, CLAHE, thresholding)."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Adaptive Thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Find contours to detect the main document area
        contours, _ = cv2.findContours(
            adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if not contours:
            raise ValueError("No document boundary found!")

        x, y, w, h = cv2.boundingRect(contours[0])
        roi = image[y : y + h, x : x + w]
        gray_roi = gray[y : y + h, x : x + w]

        # Convert back to BGR (PaddleOCR expects color images)
        gray_roi_bgr = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
        return roi, gray_roi_bgr

# -------------------- Table Extraction Class --------------------
class TableExtractor:
    """Handles text extraction and structuring into rows and columns."""

    @staticmethod
    def structure_text(bounding_boxes: List[OCRResult]) -> List[List[str]]:
        """Organize OCR-extracted text into a structured table format."""
        if not bounding_boxes:
            return []

        # Step 1: Sort bounding boxes by y-coordinate (top to bottom)
        bounding_boxes.sort(key=lambda box: box.bbox[1])

        # Step 2: Group into rows based on y-coordinates
        rows, current_row = [], [bounding_boxes[0]]
        for i in range(1, len(bounding_boxes)):
            prev_box, curr_box = current_row[-1], bounding_boxes[i]

            # If y-coordinates are close, group them as a row
            if abs(curr_box.bbox[1] - prev_box.bbox[1]) < 15:
                current_row.append(curr_box)
            else:
                rows.append(current_row)
                current_row = [curr_box]

        rows.append(current_row)  # Append last row

        # Step 3: Sort each row by x-coordinate (left to right)
        for row in rows:
            row.sort(key=lambda box: box.bbox[0])

        # Visualize the boxes

        # Step 4: Extract structured text
        return [[box.text for box in row] for row in rows],rows

# -------------------- Main Pipeline Class --------------------
class ReportProcessor:
    """Manages the complete OCR pipeline for multiple images."""

    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.ocr = OCRProcessor()

    def process_all_reports(self):
        """Process all images in the input folder."""
        image_paths = FileHandler.get_images_from_folder(self.input_folder)
        if not image_paths:
            print("No images found in the input folder!")
            return

        for image_path in image_paths:
            try:
                print(f"Processing: {image_path}")
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                sub_output_folder = os.path.join(self.output_folder, file_name.split('_')[0])

                # Preprocess image
                roi, preprocessed_image = ImagePreprocessor.preprocess_image(image_path)

                # Perform OCR
                bounding_boxes = self.ocr.extract_text_from_image(preprocessed_image)

                # Structure the text into a table
                structured_text,rows = TableExtractor.structure_text(bounding_boxes)

                # Visualize the boxes
                #self.ocr.visualize_bounding_boxes(preprocessed_image, rows)

                # Save results
                FileHandler.save_to_csv(structured_text, sub_output_folder, f"extracted_text_{file_name.split('_')[2]}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

# -------------------- Run the Script --------------------
if __name__ == "__main__":
    # conver the input pdf to single page pdf and then to image
    input_folder=DEFAULT.input
    split_pdf(input_folder,DEFAULT.single_page_pdf)
    save_pages_as_images(DEFAULT.single_page_pdf,DEFAULT.input_reports,[0])
    # Set this to the path of the final location of reports
    output_folder = DEFAULT.output_reports
    # Obtain outputs
    processor = ReportProcessor(DEFAULT.input_reports, output_folder)
    processor.process_all_reports()
    #Delete intermediate files
    FileHandler.delete_folder_securely(DEFAULT.single_page_pdf)
    FileHandler.delete_folder_securely(DEFAULT.input_reports)