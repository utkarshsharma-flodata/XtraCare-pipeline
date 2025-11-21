#!/usr/bin/env python3
"""
Dynamic Document Intelligence Extractor using GPT-4o
Processes any number of PDFs dynamically with automatic document type detection
Generates structured feature JSONs and consolidated SPARSE Bill of Entry Checklist
"""

import os
import sys
import json
import base64
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from pdf2image import convert_from_path
import io

# Load environment variables
load_dotenv()

# Optional: Initialize tracing if utils.server is available
try:
    from utils.server import init_tracing
    init_tracing()
except ImportError:
    pass  # Skip if utils.server not available


class DocumentExtractor:
    """Handles dynamic document extraction using GPT-4o with automatic type detection"""
    
    # Document type keywords for automatic detection
    DOCUMENT_TYPE_KEYWORDS = {
        'awb': ['awb', 'airwaybill', 'air_waybill', 'waybill'],
        'invoice': ['invoice', 'inv', 'commercial_invoice'],
        'packing_list': ['packing', 'packing_list', 'packinglist', 'pl'],
        'bill_of_lading': ['bol', 'bill_of_lading', 'lading'],
        'certificate': ['certificate', 'cert', 'coo'],
    }
    
    def __init__(self, api_key: Optional[str] = None, schema_dir: Optional[str] = None):
        """
        Initialize the document extractor with OpenAI API key
        
        Args:
            api_key: OpenAI API key (if None, loads from OPENAI_API_KEY env var)
            schema_dir: Directory containing schema JSON files (optional)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.schemas = self._load_schemas() if self.schema_dir else {}
        
    def _load_schemas(self) -> Dict[str, Dict]:
        """Load JSON schema templates from schema directory"""
        schemas = {}
        
        if not self.schema_dir or not self.schema_dir.exists():
            print("⚠️  Warning: Schema directory not found or not provided")
            return schemas
        
        # Load all JSON files in schema directory
        for schema_file in self.schema_dir.glob("*.json"):
            doc_type = schema_file.stem.lower()
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schemas[doc_type] = json.load(f)
                    print(f"   📋 Loaded schema: {doc_type}")
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to load schema {schema_file.name}: {str(e)}")
        
        return schemas
    
    def detect_document_type(self, filename: str) -> str:
        """
        Automatically detect document type from filename
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Detected document type (or 'unknown' if not detected)
        """
        filename_lower = filename.lower()
        
        # Check against known keywords
        for doc_type, keywords in self.DOCUMENT_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return doc_type
        
        return 'unknown'
    
    def encode_pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to base64-encoded PNG images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of base64-encoded image strings
        """
        try:
            images = convert_from_path(pdf_path, dpi=200)
            base64_images = []
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                base64_images.append(img_b64)
            return base64_images
        except Exception as e:
            raise Exception(f"Error converting PDF to images: {str(e)}")
    
    def _get_dynamic_extraction_prompt(self, doc_type: str) -> str:
        """
        Generate dynamic extraction prompt that doesn't require predefined schema
        
        Args:
            doc_type: Type of document being processed
            
        Returns:
            Extraction prompt string
        """
        return f"""You are an expert document intelligence system specialized in extracting structured data from business documents.

**DOCUMENT TYPE:** {doc_type.upper().replace('_', ' ')}

**TASK:**
Extract every meaningful and structured data point visible in this document and organize it into a logically grouped JSON hierarchy.

**EXTRACTION GUIDELINES:**

1. **Be Comprehensive**: Extract ALL visible information including:
   - Document identifiers (numbers, dates, references)
   - Party information (shipper, consignee, buyer, seller, etc.)
   - Item/product details (descriptions, quantities, prices, codes)
   - Financial information (amounts, currencies, terms)
   - Logistics details (shipping, routing, packaging)
   - Regulatory information (tax IDs, customs codes, certifications)
   - Terms and conditions
   - Any other structured data visible in the document

2. **Organize Hierarchically**: Group related fields into logical sections such as:
   - document_info
   - shipper/seller
   - consignee/buyer
   - items
   - financial_summary
   - shipping_details
   - regulatory_info
   - etc.

3. **Handle Missing Data**: For fields that are not visible or unclear, use empty string ""

4. **Preserve Data Types**:
   - Numbers: Keep as numbers (integers or floats)
   - Dates: Extract as strings in their original format
   - Arrays: Use arrays for multiple items/entries
   - Booleans: Use true/false where applicable

5. **Extract Arrays**: For line items, packages, or any repeated structures:
   - Create an array with one object per line item
   - Include all relevant fields for each item

6. **Be Precise**: Extract exactly what you see, including:
   - Exact field names and labels
   - Complete addresses and contact information
   - All numeric values with units
   - Complete product descriptions

**OUTPUT FORMAT:**
Return ONLY valid JSON without markdown formatting or explanations. The JSON should be:
- Well-structured and hierarchical
- Complete (no truncation)
- Ready for downstream processing and checklist generation

Extract the data now:"""
    
    def extract_from_pdf(self, pdf_path: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract structured data from PDF using GPT-4o Vision with dynamic schema generation
        
        Args:
            pdf_path: Path to PDF file
            doc_type: Type of document (detected or provided)
        
        Returns:
            Extracted data as dictionary
        """
        print(f"\n📄 Processing {doc_type.upper()}: {Path(pdf_path).name}")
        
        # Convert PDF pages to images
        print("   🖼️  Converting PDF to images...")
        try:
            image_base64_list = self.encode_pdf_to_images(pdf_path)
            print(f"   ✅ Converted {len(image_base64_list)} page(s)")
        except Exception as e:
            raise Exception(f"Failed to convert PDF to images: {str(e)}")
        
        # Create dynamic extraction prompt
        prompt = self._get_dynamic_extraction_prompt(doc_type)
        
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
                        "content": "You are an expert document intelligence system that extracts structured data from business documents with high accuracy and completeness."
                    },
                    {"role": "user", "content": user_content}
                ],
                max_tokens=8000,
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
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Error: Failed to parse JSON response from GPT-4o")
            print(f"      {str(e)}")
            raise
        except Exception as e:
            print(f"   ❌ Error during extraction: {str(e)}")
            raise
    
    def save_features(self, data: Dict, doc_type: str, filename: str, results_dir: str) -> str:
        """
        Save extracted features to JSON file
        
        Args:
            data: Extracted feature data
            doc_type: Type of document
            filename: Original filename
            results_dir: Directory to save results
        
        Returns:
            Path to saved file
        """
        # Create features directory
        features_dir = Path(results_dir) / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        base_name = Path(filename).stem
        output_filename = f"{base_name}_features.json"
        output_path = features_dir / output_filename
        
        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Features saved to: {output_path}")
        return str(output_path)
    
    def generate_checklist_from_features(self, 
                                        feature_files: List[str],
                                        checklist_output_dir: str,
                                        shipment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate SPARSE Bill of Entry Checklist from multiple feature JSONs using GPT-4o
        Only includes fields that were actually detected in the documents (no empty/placeholder fields)
        
        Args:
            feature_files: List of paths to feature JSON files
            checklist_output_dir: Directory to save checklist
            shipment_id: Optional shipment identifier for output filename
        
        Returns:
            Generated sparse checklist as dictionary
        """
        print(f"\n📋 Generating compact checklist (fields found only) from {len(feature_files)} document(s)...")
        
        # Load all feature JSONs
        features_data = []
        for feature_file in feature_files:
            try:
                with open(feature_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    filename = Path(feature_file).name
                    features_data.append({
                        'filename': filename,
                        'data': data
                    })
                    print(f"   📥 Loaded: {filename}")
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to load {feature_file}: {str(e)}")
        
        if not features_data:
            raise ValueError("No feature files could be loaded")
        
        # Load checklist schema if available (for reference only)
        checklist_schema = self.schemas.get('checklist', {})
        schema_str = json.dumps(checklist_schema, indent=2) if checklist_schema else "No predefined schema available"
        
        # Build comprehensive prompt for SPARSE checklist generation
        prompt = f"""
You are a customs documentation system specialized in generating Bill of Entry checklists.

**CRITICAL INSTRUCTIONS – READ CAREFULLY**

Your task is to generate a **sparse but structurally complete JSON checklist** based on the extracted features from multiple documents.

**DOCUMENT COUNT:** {len(features_data)}

---

### STRUCTURE RULES

1. **Preserve All Sections and Items**
   - Always include every section (document_info, importer_details, shipment_details, item_details, duty_summary, etc.) that was detected in any source document.
   - Always include **all detected items** inside `item_details`, even if some of their fields are missing.
   - If an item exists with partial data (e.g., only description and HS code), it must still appear as a separate object in the `item_details` array.

2. **Include Only Available Fields**
   - Within each section or item, include only the fields that have actual values.
   - Do not insert placeholder, null, or empty fields.
   - Do not generate keys that have no detected value.

3. **Maintain Schema Compatibility**
   - Use key names and nesting similar to the reference schema so this JSON can be merged directly later.
   - Use consistent naming conventions (e.g., "item_description", "hs_code", "uom", "quantity", "invoice_value", etc.) from the schema below.

4. **Consolidate Intelligently**
   - Merge overlapping information from different documents.
   - Prefer the most detailed or specific version when conflicts occur.
   - Ensure internal consistency (e.g., total quantities, weights, and amounts should align).

5. **Output Format**
   - Output valid JSON only — no markdown, comments, or text outside the JSON.
   - The JSON must be compact and hierarchical.
   - Each array (especially `item_details`) should contain one object per detected item.

---

### CHECKLIST SCHEMA (REFERENCE ONLY – DO NOT EXPAND FULLY)
```json
{schema_str}


**SOURCE DOCUMENTS AND EXTRACTED FEATURES:**
"""
        
        # Add each document's features to the prompt
        for idx, feature_doc in enumerate(features_data, 1):
            prompt += f"""

**DOCUMENT {idx}: {feature_doc['filename']}**
```json
{json.dumps(feature_doc['data'], indent=2)}
```
"""
        
        prompt += """

FINAL INSTRUCTIONS

Now generate the sparse but structurally complete checklist JSON:

Include all detected sections and items.

Include only the keys that have actual values.

Preserve array structure (one item object per detected item).

Ensure valid, parsable JSON output with no markdown or commentary."""

        # Call GPT-4o to generate sparse checklist
        print("   🤖 Calling GPT-4o to generate compact checklist...")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert customs documentation system that creates compact, sparse Bill of Entry checklists by intelligently consolidating shipping documents. You ONLY include fields that were actually found in the source documents, with no empty or placeholder values."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=8000,
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
            
            sparse_checklist_data = json.loads(content)
            print("   ✅ Compact checklist generation completed successfully")
            
            # Save sparse checklist
            output_path = self._save_sparse_checklist(sparse_checklist_data, checklist_output_dir, shipment_id)
            
            return sparse_checklist_data
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Error: Failed to parse JSON response from GPT-4o")
            print(f"      {str(e)}")
            raise
        except Exception as e:
            print(f"   ❌ Error during checklist generation: {str(e)}")
            raise
    
    def _save_sparse_checklist(self, checklist_data: Dict, output_dir: str, shipment_id: Optional[str] = None) -> str:
        """
        Save sparse checklist to output directory
        
        Args:
            checklist_data: Sparse checklist dictionary
            output_dir: Directory to save checklist
            shipment_id: Optional shipment identifier
        
        Returns:
            Path to saved checklist
        """
        # Create checklist directory
        checklist_dir = Path(output_dir) / "checklists"
        checklist_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename with _partial_checklist suffix
        if shipment_id:
            output_filename = f"{shipment_id}_partial_checklist.json"
        else:
            output_filename = "unnamed_partial_checklist.json"
        
        output_path = checklist_dir / output_filename
        
        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(checklist_data, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 Saved partial checklist: {output_path}")
        return str(output_path)
    
    def _save_full_checklist(self, checklist_data: Dict, output_dir: str, shipment_id: Optional[str] = None) -> str:
        """
        Save merged full checklist to output directory
        """
        checklist_dir = Path(output_dir) / "checklists"
        checklist_dir.mkdir(parents=True, exist_ok=True)

        if shipment_id:
            output_filename = f"{shipment_id}_checklist.json"
        else:
            output_filename = "unnamed_checklist.json"

        output_path = checklist_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(checklist_data, f, indent=2, ensure_ascii=False)

        return str(output_path)

    
    def merge_sparse_to_schema(self, sparse_json: Dict[str, Any], schema_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge a sparse checklist JSON into the full schema structure.
        Ensures that no sections are skipped and arrays always preserve schema structure,
        even when the sparse JSON provides an empty list.
        """

        def merge_recursive(schema_node: Any, sparse_node: Any) -> Any:
            # 1️⃣ Handle dict nodes
            if isinstance(schema_node, dict):
                result = {}
                for key, schema_value in schema_node.items():
                    sparse_value = None
                    if isinstance(sparse_node, dict):
                        sparse_value = sparse_node.get(key)

                    if sparse_value is not None:
                        result[key] = merge_recursive(schema_value, sparse_value)
                    else:
                        result[key] = merge_recursive(schema_value, None)
                return result

            # 2️⃣ Handle list nodes
            elif isinstance(schema_node, list):
                # Schema defines a list of dicts (e.g., item_details, igst_details)
                if len(schema_node) > 0 and isinstance(schema_node[0], dict):
                    if isinstance(sparse_node, list) and len(sparse_node) > 0:
                        # Merge each detected item
                        return [merge_recursive(schema_node[0], item) for item in sparse_node]
                    else:
                        # If sparse is empty, still include one empty dict based on schema
                        return [merge_recursive(schema_node[0], {})]
                else:
                    # Simple list (like strings or values)
                    if isinstance(sparse_node, list) and len(sparse_node) > 0:
                        return sparse_node
                    else:
                        return []

            # 3️⃣ Handle leaf values
            else:
                if sparse_node is not None:
                    return sparse_node
                elif isinstance(schema_node, (int, float)):
                    return 0
                elif isinstance(schema_node, bool):
                    return False
                elif isinstance(schema_node, str):
                    return ""
                else:
                    return ""

        # Perform recursive merge
        return merge_recursive(schema_json, sparse_json)

    def process_documents_dynamic(self, 
                                  pdf_paths: List[str], 
                                  results_dir: str, 
                                  checklist_output_dir: str,
                                  shipment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main dynamic processing pipeline - processes any number of PDFs and generates sparse checklist
        
        Args:
            pdf_paths: List of paths to PDF files to process
            results_dir: Directory to save extracted feature JSONs
            checklist_output_dir: Directory to save final sparse checklist
            shipment_id: Optional shipment identifier for output files
        
        Returns:
            Dictionary containing processing results and paths
        """
        print("=" * 70)
        print("  📦 DYNAMIC DOCUMENT INTELLIGENCE PIPELINE (SPARSE MODE)")
        print("=" * 70)
        print(f"\n📋 Processing {len(pdf_paths)} document(s)")
        print(f"📁 Results directory: {results_dir}")
        print(f"📁 Checklist directory: {checklist_output_dir}")
        if shipment_id:
            print(f"🆔 Shipment ID: {shipment_id}")
        
        results = {
            'documents': {},
            'feature_files': [],
            'checklist': None,
            'summary': {
                'total': len(pdf_paths),
                'success': 0,
                'failed': 0,
                'skipped': 0
            }
        }
        
        # Step 1: Process each PDF
        for pdf_path in pdf_paths:
            pdf_path = str(pdf_path)  # Ensure string type
            filename = Path(pdf_path).name
            
            # Check if file exists
            if not Path(pdf_path).exists():
                print(f"\n❌ File not found: {pdf_path}")
                results['documents'][filename] = {
                    'status': 'failed',
                    'error': 'File not found'
                }
                results['summary']['failed'] += 1
                continue
            
            try:
                # Detect document type
                doc_type = self.detect_document_type(filename)
                print(f"\n🔍 Detected document type: {doc_type}")
                
                # Extract features
                extracted_data = self.extract_from_pdf(pdf_path, doc_type)
                
                # Save features
                feature_path = self.save_features(
                    extracted_data, 
                    doc_type, 
                    filename, 
                    results_dir
                )
                
                # Record success
                results['documents'][filename] = {
                    'status': 'success',
                    'doc_type': doc_type,
                    'feature_path': feature_path
                }
                results['feature_files'].append(feature_path)
                results['summary']['success'] += 1
                
            except Exception as e:
                print(f"\n❌ Failed to process {filename}: {str(e)}")
                results['documents'][filename] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['summary']['failed'] += 1
        
        # Step 2: Generate sparse checklist if at least one document succeeded
        if results['summary']['success'] > 0:
            print(f"\n{'=' * 70}")
            try:
                sparse_checklist_data = self.generate_checklist_from_features(
                    results['feature_files'],
                    checklist_output_dir,
                    shipment_id
                )
                
                results['checklist'] = {
                    'status': 'success',
                    'data': sparse_checklist_data
                }
                
            except Exception as e:
                print(f"\n❌ Failed to generate checklist: {str(e)}")
                results['checklist'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        else:
            print(f"\n⚠️  Skipping checklist generation - no documents processed successfully")
            results['checklist'] = {
                'status': 'skipped',
                'error': 'No documents processed successfully'
            }
        
        # Step 2.5: Merge sparse checklist into full schema and save final checklist
        if 'checklist' in results and results['checklist']['status'] == 'success':
            print("\n🧩 Merging sparse checklist into full schema...")

            checklist_schema = self.schemas.get('checklist', {})
            if checklist_schema:
                merged_checklist = self.merge_sparse_to_schema(
                    sparse_json=results['checklist']['data'],
                    schema_json=checklist_schema
                )

                # Save merged checklist
                merged_path = self._save_full_checklist(
                    merged_checklist,
                    checklist_output_dir,
                    shipment_id
                )
                print(f"   💾 Saved merged full checklist: {merged_path}")
                results['checklist']['merged_path'] = merged_path
            else:
                print("   ⚠️ No checklist schema found — skipping merge step.")


        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print processing summary"""
        print("\n" + "=" * 70)
        print("  📊 PROCESSING SUMMARY")
        print("=" * 70)
        
        summary = results['summary']
        print(f"\n📈 Total Documents: {summary['total']}")
        print(f"   ✅ Success: {summary['success']}")
        print(f"   ❌ Failed: {summary['failed']}")
        if summary['skipped'] > 0:
            print(f"   ⏭️  Skipped: {summary['skipped']}")
        
        print("\n📄 Document Details:")
        for filename, doc_result in results['documents'].items():
            status_icon = "✅" if doc_result['status'] == 'success' else "❌"
            print(f"   {status_icon} {filename}")
            if doc_result['status'] == 'success':
                print(f"      Type: {doc_result['doc_type']}")
                print(f"      Features: {doc_result['feature_path']}")
            else:
                print(f"      Error: {doc_result.get('error', 'Unknown error')}")
        
        print("\n📋 Checklist Status:")
        checklist = results['checklist']
        if checklist:
            if checklist['status'] == 'success':
                print("   ✅ Sparse checklist generated successfully")
            elif checklist['status'] == 'skipped':
                print(f"   ⏭️  Skipped: {checklist.get('error', 'Unknown reason')}")
            else:
                print(f"   ❌ Failed: {checklist.get('error', 'Unknown error')}")
        
        print("=" * 70)
        
        if summary['failed'] == 0 and checklist.get('status') == 'success':
            print("\n✨ Pipeline completed successfully!")
        elif summary['success'] > 0:
            print(f"\n⚠️  Pipeline completed with {summary['failed']} failure(s)")
        else:
            print("\n❌ Pipeline failed - no documents processed successfully")


def main():
    """
    Example usage of the dynamic pipeline
    Modify the pdf_paths list to process your documents
    """
    
    # ============================================================================
    # CONFIGURATION: Edit these paths for your use case
    # ============================================================================
    
    # List of PDF paths to process (can be any number of documents)
    pdf_paths = [
        r"D:\XtraLogistics\data\SHIPMENT DETAILS\SA250910843\SA25091084 HAWB.pdf",
        r"D:\XtraLogistics\data\SHIPMENT DETAILS\SA250910843\SA25091084 INV.pdf",
        r"D:\XtraLogistics\data\SHIPMENT DETAILS\SA250910843\SA25091084 MAWB.pdf"
    ]
    
    # Output directories
    results_dir = "results/SA25091084"  # Will create results/features/ subdirectory
    checklist_output_dir = "results/SA25091084"  # Will create results/checklists/ subdirectory
    
    # Optional: Shipment ID for naming the checklist
    shipment_id = "SA25091084"  # or None for default naming
    
    # Optional: Schema directory (if you have predefined schemas)
    schema_dir = r"D:\XtraLogistics\schema"  # or None if not using schemas
    
    # Optional: API key (leave None to use .env file)
    api_key = None
    
    # ============================================================================
    # EXECUTION
    # ============================================================================
    
    try:
        # Initialize extractor
        extractor = DocumentExtractor(api_key=api_key, schema_dir=schema_dir)
        
        # Run dynamic processing pipeline
        results = extractor.process_documents_dynamic(
            pdf_paths=pdf_paths,
            results_dir=results_dir,
            checklist_output_dir=checklist_output_dir,
            shipment_id=shipment_id
        )
        
        # Exit with appropriate code
        if results['summary']['failed'] == 0 and results['checklist'].get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()