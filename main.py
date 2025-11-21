import fitz
import re
import json

pdf_path = "AirwayBill2.pdf"

# define bounding boxes for each field
regions = {
    "MAWB_number": (30, 30, 165, 45),
    "HAWB_number": (375, 30, 475, 45),
    "Shipper_name_and_address": (30, 50, 280, 100),
    "Consignee_name_and_address": (30, 130, 260, 180),
    "Airport_of_departure": (31, 275, 80, 285),
    "Airport_of_destination": (31, 310, 120, 340),
    "No_of_pieces_rcp": (31.3, 407.7, 36.3, 416.7),
    "Gross_weight": (70, 380, 120, 420),
    "Chargeable_weight": (190.0, 407.7, 393.3, 416.7)
}

def extract_text_in_region(page, rect):
    """Extract text from blocks that fall within a rectangular region"""
    x0, y0, x1, y1 = rect
    text = []
    for bx0, by0, bx1, by1, btext, *_ in page.get_text("blocks"):
        if (bx1 >= x0 and bx0 <= x1) and (by1 >= y0 and by0 <= y1):
            text.append(btext.strip())
    return " ".join(text)

def clean_extracted_data(data: dict) -> dict:
    """
    Cleans the extracted text values in a JSON-like dictionary.
    Removes unwanted labels, duplicate field names, and extra words.
    """
    clean_map = {}

    for key, val in data.items():
        if not val:
            clean_map[key] = None
            continue

        # Normalize whitespace
        text = " ".join(val.split())

        # Common field labels to remove
        remove_words = [
            "Shipper's Name and Address",
            "Consignee's Name and Address",
            "Shipper's Account Number",
            "MAWB number", "HAWB number",
            "Airport of Destination",
            "Airport of Departure",
            "Gross Weight", "Chargeable Weight",
            "Handling Information", "Change", "kg", "lb", "Rate", "Total"
        ]
        for w in remove_words:
            text = re.sub(rf"\b{re.escape(w)}\b", "", text, flags=re.IGNORECASE)

        # Cleanup multiple spaces, punctuation artifacts
        text = re.sub(r"\s{2,}", " ", text).strip(" ,;:-")

        # Field-specific tweaks
        if key == "MAWB_number":
            # Keep only digits and letters separated by space
            text = re.sub(r"[^A-Za-z0-9\s]", "", text)
            text = " ".join(re.findall(r"[A-Za-z0-9]+", text))
        elif key == "HAWB_number":
            text = re.sub(r"[^A-Za-z0-9]", "", text)
        elif key in ("Gross_weight", "Chargeable_weight"):
            # keep only numbers, K, or decimal parts
            text = " ".join(re.findall(r"[A-Za-z0-9\.\-]+", text))
        elif key in ("Airport_of_departure", "Airport_of_destination"):
            # likely one word like "SHENZHEN" or "NEW DELHI"
            parts = text.split()
            text = " ".join(parts[-2:]) if len(parts) > 2 else text

        clean_map[key] = text.strip()

    return clean_map

doc = fitz.open(pdf_path)
page = doc[0]
result = {}

for key, rect in regions.items():
    val = extract_text_in_region(page, rect)
    # clean up line breaks and multiple spaces
    val = " ".join(val.split())
    result[key] = val if val else None

doc.close()

cleaned = clean_extracted_data(result)
print(json.dumps(cleaned, indent=2, ensure_ascii=False))
