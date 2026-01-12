#extractor/step1_ingest.py
import re
from datetime import datetime
from pathlib import Path
from step1_pdf_utils import pdf_to_images
from step1_ocr_utils import ocr_image_to_text, ocr_image_to_data
from param_extractor import extract_lab_parameters_from_ocr_data, extract_lab_parameters_from_text
from json_utils import load_json_file


def extract_patient_info_from_text(text):
    """
    Attempt to extract patient_id, age, and gender from the text using regex patterns.
    """
    patient_id, age, gender = None, None, None
    pid_match = re.search(r"Patient\s*ID\s*[:\-]?\s*(\S+)", text, re.IGNORECASE)
    if pid_match:
        patient_id = pid_match.group(1).strip()
    age_match = re.search(r"Age\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    if age_match:
        age = age_match.group(1).strip()
    gender_match = re.search(r"Gender\s*[:\-]?\s*(Male|Female|Other)", text, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1).strip().capitalize()
    return patient_id, age, gender

def process_pdf_file(filepath):
    """
    Process a PDF: convert pages to images, OCR each page, extract parameters and patient info.
    """
    images = pdf_to_images(filepath)
    all_text_pages = []
    all_params = []
    for i, img in enumerate(images, start=1):
        text = ocr_image_to_text(img)
        all_text_pages.append(text)
        ocr_data = ocr_image_to_data(img)
        params = extract_lab_parameters_from_ocr_data(ocr_data)
        # Fallback to line-based extraction if none found
        if not params:
            params = extract_lab_parameters_from_text(text, page=i)
        all_params.extend(params)
    combined_text = "\n".join(all_text_pages)
    patient_id, age, gender = extract_patient_info_from_text(combined_text)
    return patient_id, age, gender, all_params

def process_image_file(filepath):
    """
    Process an image file: OCR, then extract parameters and patient info.
    """
    from PIL import Image
    img = Image.open(filepath)
    text = ocr_image_to_text(img)
    ocr_data = ocr_image_to_data(img)
    params = extract_lab_parameters_from_ocr_data(ocr_data)
    if not params:
        params = extract_lab_parameters_from_text(text, page=1)
    patient_id, age, gender = extract_patient_info_from_text(text)
    return patient_id, age, gender, params

def process_json_file(filepath):
    """
    Process a JSON input.
    Supports:
      1) Extractor-generated JSON with `extracted`
      2) External blood-report JSON with `parameters`
    """
    from json_utils import load_json
    from param_extractor import normalize_number, fuzzy_canonical

    data = load_json(str(filepath))

    patient_id = data.get("patient_id")
    age = data.get("age")
    gender = data.get("gender")

    extracted = []

    # Case 1: Already extractor-style
    if isinstance(data.get("extracted"), list):
        extracted = data["extracted"]

    # Case 2: External blood report JSON
    elif isinstance(data.get("parameters"), dict):
        for raw_name, raw_value in data["parameters"].items():
            value = normalize_number(raw_value)
            canonical, score = fuzzy_canonical(raw_name)

            extracted.append({
                "raw_name": raw_name,
                "raw_value": raw_value,
                "value": value,
                "unit": None,
                "canonical": canonical,
                "match_confidence": score,
                "source": "json_parameters",
                "raw_row_text": None,
                "value_confidence": "high" if value is not None else "unknown",
                "suspect_reason": None
            })

    return patient_id, age, gender, extracted

def ingest_file(input_path):
    """
    Main ingestion function: detect input type and process accordingly.
    Returns a dict ready for JSON output with patient info, extracted params, and meta.
    """
    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    if ext == '.pdf':
        source_type = 'pdf'
        patient_id, age, gender, extracted = process_pdf_file(str(input_path))
    elif ext in ['.jpg', '.jpeg', '.png', '.tiff']:
        source_type = 'image'
        patient_id, age, gender, extracted = process_image_file(str(input_path))
    elif ext == '.json':
        source_type = 'json'
        patient_id, age, gender, extracted = process_json_file(str(input_path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    result = {
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'extracted': extracted,
        'meta': {
            'ingestion_timestamp': datetime.utcnow().isoformat() + 'Z',
            'source_type': source_type,
            'notes': f"Ingested {input_path.name}"
        }
    }
    return result
