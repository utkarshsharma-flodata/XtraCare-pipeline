#!/usr/bin/env python3
"""
Document Intelligence Extractor using GPT-4o
Extracts structured data from AWB, Invoice, and Packing List PDFs
Then generates a Bill of Entry Checklist by merging the extracted data
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

from pdf2image import convert_from_path
import io

# Load environment variables
load_dotenv()

from utils.server import init_tracing
init_tracing()


class DocumentExtractor:
    """Handles document extraction using GPT-4o"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the document extractor with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        self.schemas = self._load_schemas()
        
    def _load_schemas(self) -> Dict[str, Dict]:
        """Load JSON schema templates"""
        schema_dir = Path(__file__).parent / "schemas"
        schemas = {}
        
        schema_files = {
            "awb": r"D:\XtraLogistics\schema\AWB.json",
            "invoice": r"D:\XtraLogistics\schema\invoice.json",
            "packing_list": r"D:\XtraLogistics\schema\packingList.json",
            "checklist": r"D:\XtraLogistics\schema\checklist.json"  # Added checklist schema
        }
        
        for doc_type, filename in schema_files.items():
            schema_path = schema_dir / filename
            if schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schemas[doc_type] = json.load(f)
            else:
                print(f"⚠️  Warning: Schema file {filename} not found at {schema_path}")
                schemas[doc_type] = {}
        
        return schemas
    
    def encode_pdf(self, pdf_path: str) -> str:
        """Encode PDF file to base64 string"""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                return base64.b64encode(pdf_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        except Exception as e:
            raise Exception(f"Error encoding PDF {pdf_path}: {str(e)}")
    
    def _get_extraction_prompt(self, doc_type: str, schema: Dict) -> str:
        """Generate detailed extraction prompt for specific document type"""
        
        prompts = {
            "awb": """You are an expert in Air Waybill (AWB) document analysis. Extract ALL information from this AWB PDF document and structure it according to the provided JSON schema.

**EXTRACTION GUIDELINES:**
1. Extract EVERY field visible in the document, even if values are partially visible
2. For missing or unclear fields, use empty string ""
3. Maintain exact field names from the schema
4. Parse dates in the format found in the document (e.g., DD-MMM-YYYY, MM/DD/YYYY)
5. For addresses, combine all address lines into a single string
6. Extract all numeric values with their units (e.g., "100 KG", "50 pieces")
7. For routing information, extract complete flight paths and carrier codes

**KEY SECTIONS TO EXTRACT:**
- Document Info: HAWB/MAWB numbers, dates, issuing details
- Shipper: Name, address, contact details, account numbers
- Consignee: Complete delivery address and contact information
- Notify Party: Additional notification contact with GSTIN/CIN if present
- Routing: Origin, destination, carrier, flight numbers, dates
- Cargo: Pieces, weight, dimensions, HS codes, goods description
- Charges: All freight charges, currency, prepaid/collect amounts
- Carrier & Certification: Carrier details and dangerous goods declarations

Return ONLY valid JSON matching the schema structure. Do not include markdown formatting or explanations.""",

            "invoice": """You are an expert in commercial invoice document analysis. Extract ALL information from this Invoice PDF and structure it according to the provided JSON schema.

**EXTRACTION GUIDELINES:**
1. Extract EVERY field visible in the document with precision
2. For missing fields, use empty string ""
3. Maintain exact field names from the schema
4. Parse dates accurately
5. Extract all monetary values with currency symbols
6. For item details, create a list entry for EACH line item
7. Calculate totals if not explicitly stated

**KEY SECTIONS TO EXTRACT:**
- Document Info: Invoice number, date, currency, customer ID
- Seller: Complete company details including address and contact
- Buyer: Full buyer information and contact details
- Ship To: Shipping destination with estimated dates and weights
- Manufacturer: If different from seller, extract manufacturer details
- Item Details: For EACH item extract:
  * Part/Product number
  * Description
  * HS Code
  * Quantity and unit of measure
  * Unit price and total amount
- Financial Summary: Subtotal, taxes, freight, insurance, other charges, grand total
- Terms: Incoterms, payment terms, additional instructions

Return ONLY valid JSON matching the schema structure. Do not include markdown formatting or explanations.""",

            "packing_list": """You are an expert in packing list document analysis. Extract ALL information from this Packing List PDF and structure it according to the provided JSON schema.

**EXTRACTION GUIDELINES:**
1. Extract EVERY field visible in the document
2. For missing fields, use empty string ""
3. Maintain exact field names from the schema
4. For item details, create a list entry for EACH product line
5. Extract all package/carton information with quantities
6. Capture complete company registration details

**KEY SECTIONS TO EXTRACT:**
- Document Info: Document/order numbers, dates, delivery terms, references
- Delivery Details: Complete shipping destination address and contact
- Invoice Details: Billing address if different from delivery
- Item Details: For EACH item extract:
  * Product number
  * Description
  * Quantity
  * Unit of measure
- Shipment Details: Pickup location, freight company, shipping address
- Company Information: Complete seller details including:
  * Corporate ID, VAT, EORI numbers
  * Banking details (SWIFT, IBAN)
  * Business register information
  * Legal notes

Return ONLY valid JSON matching the schema structure. Do not include markdown formatting or explanations."""
        }
        
        base_prompt = prompts.get(doc_type, "Extract all information from this document.")
        schema_str = json.dumps(schema, indent=2)
        
        return f"""{base_prompt}

**JSON SCHEMA TO FOLLOW:**
```json
{schema_str}
```

Extract the data and return ONLY the JSON object. Ensure all top-level keys from the schema are present."""
    
    def encode_pdf_to_images(self, pdf_path: str):
        """Convert PDF pages to base64-encoded PNG images"""
        images = convert_from_path(pdf_path, dpi=200)
        base64_images = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_b64)
        return base64_images

    
    def extract_from_pdf(self, pdf_path: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract structured data from PDF using GPT-4o Vision
        
        Args:
            pdf_path: Path to PDF file
            doc_type: Type of document ('awb', 'invoice', 'packing_list')
        
        Returns:
            Extracted data as dictionary
        """
        print(f"\n📄 Processing {doc_type.upper()}: {Path(pdf_path).name}")
        
        # Get schema for this document type
        schema = self.schemas.get(doc_type, {})
        if not schema:
            raise ValueError(f"Schema not found for document type: {doc_type}")
        
        # Convert PDF pages to images
        print("   🖼️  Converting PDF to images...")
        image_base64_list = self.encode_pdf_to_images(pdf_path)
        
        # Create extraction prompt
        prompt = self._get_extraction_prompt(doc_type, schema)

        # Build GPT-4o message with multiple image inputs
        image_inputs = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            }
            for img_b64 in image_base64_list
        ]
        
        # Combine text prompt + all images
        user_content = [{"type": "text", "text": prompt}] + image_inputs

        # Call GPT-4o with vision capabilities
        print("   🤖 Calling GPT-4o for extraction...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document intelligence system..."
                    },
                    {"role": "user", "content": user_content}
                ],
                max_tokens=4096,
                temperature=0.1,
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            extracted_data = json.loads(content)
            print("   ✅ Extraction completed successfully")
            
            # Validate schema compliance
            self._validate_schema(extracted_data, schema, doc_type)
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Error: Failed to parse JSON response from GPT-4o")
            print(f"      {str(e)}")
            raise
        except Exception as e:
            print(f"   ❌ Error during extraction: {str(e)}")
            raise
    
    def _validate_schema(self, data: Dict, schema: Dict, doc_type: str) -> None:
        """Validate that extracted data matches schema structure"""
        print("   🔍 Validating schema compliance...")
        
        missing_keys = []
        schema_keys = set(schema.keys())
        data_keys = set(data.keys())
        
        missing_keys = schema_keys - data_keys
        
        if missing_keys:
            print(f"   ⚠️  Warning: Missing top-level keys in {doc_type}: {missing_keys}")
        else:
            print("   ✅ All schema keys present")
    
    def save_output(self, data: Dict, doc_type: str, original_filename: str) -> str:
        """Save extracted data to JSON file in appropriate results directory"""
        
        # Create results directory structure
        results_dir = Path("results") / doc_type
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        base_name = Path(original_filename).stem
        output_filename = f"{base_name}_output.json"
        output_path = results_dir / output_filename
        
        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Saved to: {output_path}")
        return str(output_path)
    
    def create_checklist(self, awb_data: Dict, invoice_data: Dict, packing_data: Dict, 
                        checklist_schema: Dict, base_filename: str) -> Dict[str, Any]:
        """
        Create Bill of Entry Checklist by merging data from three documents using GPT-4o
        
        Args:
            awb_data: Extracted AWB data
            invoice_data: Extracted Invoice data
            packing_data: Extracted Packing List data
            checklist_schema: Target schema for checklist
            base_filename: Base filename for output
        
        Returns:
            Generated checklist as dictionary
        """
        print(f"\n📋 Generating Bill of Entry Checklist...")
        
        # Build the prompt for checklist generation
        schema_str = json.dumps(checklist_schema, indent=2)
        
        prompt = f"""You are a document intelligence system specialized in customs documentation.

**TASK:**
Combine the extracted data from three shipping documents to generate a comprehensive Bill of Entry Checklist.

**SOURCE DOCUMENTS:**
1. **Air Waybill (AWB)** - Contains shipping details, routing, cargo information
2. **Commercial Invoice** - Contains item details, pricing, parties involved
3. **Packing List** - Contains package details, item quantities, company information

**INSTRUCTIONS:**
1. Analyze all three JSON documents provided below
2. Merge and consolidate information intelligently:
   - Use the most complete/detailed information when data appears in multiple sources
   - Cross-reference to ensure consistency (e.g., quantities, weights, descriptions)
   - Prioritize Invoice data for financial/commercial details
   - Prioritize AWB for routing and shipment details
   - Prioritize Packing List for packaging and company registration details
3. Generate a single consolidated JSON following the checklist schema exactly
4. Fill ALL keys where data is available from any of the three sources
5. For fields not present in any source, use empty string ""
6. Ensure all calculations are accurate (totals, weights, quantities)
7. Maintain data integrity and traceability

**CHECKLIST SCHEMA:**
```json
{schema_str}
```

**SOURCE DATA:**

**AIR WAYBILL DATA:**
```json
{json.dumps(awb_data, indent=2)}
```

**INVOICE DATA:**
```json
{json.dumps(invoice_data, indent=2)}
```

**PACKING LIST DATA:**
```json
{json.dumps(packing_data, indent=2)}
```

Generate the complete Bill of Entry Checklist JSON. Return ONLY the JSON object without markdown formatting or explanations."""

        # Call GPT-4o to generate checklist
        print("   🤖 Calling GPT-4o to merge documents...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert customs documentation system that creates accurate Bill of Entry checklists by intelligently merging shipping documents."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4096,
                temperature=0.1,
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            checklist_data = json.loads(content)
            print("   ✅ Checklist generation completed successfully")
            
            # Validate schema compliance
            self._validate_schema(checklist_data, checklist_schema, "checklist")
            
            # Save checklist
            output_path = self._save_checklist(checklist_data, base_filename)
            
            return checklist_data
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Error: Failed to parse JSON response from GPT-4o")
            print(f"      {str(e)}")
            raise
        except Exception as e:
            print(f"   ❌ Error during checklist generation: {str(e)}")
            raise
    
    def _save_checklist(self, checklist_data: Dict, base_filename: str) -> str:
        """Save checklist to results/checklist directory"""
        
        # Create checklist directory
        checklist_dir = Path("results") / "checklist"
        checklist_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{base_filename}_checklist.json"
        output_path = checklist_dir / output_filename
        
        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(checklist_data, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Checklist saved to: {output_path}")
        return str(output_path)


def main():
    """Main execution function (with editable input paths inside the script)"""

    # 📝 Manually set your input file paths here
    awb_path = r"D:\XtraLogistics\data\shipment2810\SHIPMENT DETAILS\93600908224 awb.pdf"
    invoice_path = r"D:\XtraLogistics\data\shipment2810\SHIPMENT DETAILS\93600908224 inv.pdf"
    packing_list_path = r"D:\XtraLogistics\data\shipment2810\SHIPMENT DETAILS\93600908224 pl.pdf"

    # Optionally: set your API key here (or leave None to use .env)
    api_key = None  # e.g., "sk-xxxxx"

    print("=" * 70)
    print("  📦 DOCUMENT INTELLIGENCE EXTRACTOR - GPT-4o")
    print("=" * 70)

    try:
        # Initialize extractor
        extractor = DocumentExtractor(api_key=api_key)

        # Process documents
        documents = [
            (awb_path, 'awb'),
            (invoice_path, 'invoice'),
            (packing_list_path, 'packing_list')
        ]

        results = {}
        extracted_data = {}

        # Step 1-3: Extract individual documents
        for pdf_path, doc_type in documents:
            try:
                # Extract data
                data = extractor.extract_from_pdf(pdf_path, doc_type)
                extracted_data[doc_type] = data

                # Save output
                output_path = extractor.save_output(data, doc_type, pdf_path)
                results[doc_type] = {
                    'status': 'success',
                    'output_path': output_path
                }

            except Exception as e:
                print(f"\n❌ Failed to process {doc_type}: {str(e)}")
                results[doc_type] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # Step 4: Generate Bill of Entry Checklist (only if all extractions succeeded)
        all_succeeded = all(r['status'] == 'success' for r in results.values())
        
        if all_succeeded:
            try:
                # Get base filename from AWB path
                base_filename = Path(awb_path).stem
                
                # Get checklist schema
                checklist_schema = extractor.schemas.get('checklist', {})
                
                # Generate checklist
                checklist_data = extractor.create_checklist(
                    awb_data=extracted_data['awb'],
                    invoice_data=extracted_data['invoice'],
                    packing_data=extracted_data['packing_list'],
                    checklist_schema=checklist_schema,
                    base_filename=base_filename
                )
                
                results['checklist'] = {
                    'status': 'success',
                    'output_path': f"results/checklist/{base_filename}_checklist.json"
                }
                
            except Exception as e:
                print(f"\n❌ Failed to generate checklist: {str(e)}")
                results['checklist'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        else:
            print("\n⚠️  Skipping checklist generation due to extraction failures")
            results['checklist'] = {
                'status': 'skipped',
                'error': 'Previous extractions failed'
            }

        # Print summary
        print("\n" + "=" * 70)
        print("  📊 EXTRACTION SUMMARY")
        print("=" * 70)

        for doc_type, result in results.items():
            if result['status'] == 'success':
                status_icon = "✅"
            elif result['status'] == 'skipped':
                status_icon = "⏭️"
            else:
                status_icon = "❌"
            
            print(f"{status_icon} {doc_type.upper()}: {result['status']}")
            if result['status'] == 'success':
                print(f"   Output: {result['output_path']}")
            elif result['status'] == 'failed':
                print(f"   Error: {result.get('error', 'Unknown error')}")

        print("=" * 70)

        # Exit with appropriate code
        failed_count = sum(1 for r in results.values() if r['status'] == 'failed')
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} document(s) failed to process")
            sys.exit(1)
        else:
            print("\n✨ All documents processed successfully!")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()