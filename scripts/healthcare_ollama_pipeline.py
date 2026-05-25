"""
Healthcare Document Processing Pipeline

This pipeline integrates document classification with template-based information extraction.
It automatically:
1. Classifies the input document image
2. Selects appropriate prompt and JSON templates
3. Extracts structured information using Ollama
4. Saves the results to JSON
"""

import argparse
import json
import base64
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import requests

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
CLASSIFIER_MODEL = "qwen3.5:9b-q4_K_M"
EXTRACTOR_MODEL = "gemma4:e4b-it-bf16"  # Can be overridden

# Document type mappings from healthcare/hoso1/doctype.md
DOCUMENT_TYPES = {
    "PHIẾU KHÁM BỆNH VÀO VIỆN": {
        "json": "page-03.json",
        "template": "healthcare_types/templates_v1/page-03-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
    "PHIẾU CHỈ ĐỊNH CẬN LÂM SÀNG": {
        "json": "page-04.json",
        "template": "healthcare_types/templates_v1/page-04-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
    "PHIẾU TRẢ KẾT QUẢ HUYẾT HỌC": {
        "json": "page-23.json",
        "template": "healthcare_types/templates_v1/page-23-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
    "PHIẾU CHĂM SÓC": {
        "json": "page-24.json",
        "template": "healthcare_types/templates_v1/page-24-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
    "PHIẾU THEO DÕI CHỨC NĂNG SỐNG": {
        "json": "page-36.json",
        "template": "healthcare_types/templates_v1/page-36-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
    "TỜ ĐIỀU TRỊ": {
        "json": "page-38.json",
        "template": "healthcare_types/templates_v1/page-38-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
    "GIẤY RA VIỆN": {
        "json": "page-60.json",
        "template": "healthcare_types/templates_v1/page-60-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
    "BẢNG KÊ CHI PHÍ ĐIỀU TRỊ NGOẠI TRÚ": {
        "json": "page-61.json",
        "template": "healthcare_types/templates_v1/page-61-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
    "PHIẾU CÔNG KHAI DỊCH VỤ KCB NỘI TRÚ": {
        "json": "page-62.json",
        "template": "healthcare_types/templates_v1/page-62-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
    "BỆNH ÁN THẬN NHÂN TẠO": {
        "json": "page-01.json",
        "template": "healthcare_types/templates_v1/page-01-template.json",
        "prompt_template": "healthcare_types/templates_v1/prompt_with_template.txt"
    },
}

DOCUMENT_NAMES = list(DOCUMENT_TYPES.keys())


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def classify_document(
    image_path: str,
    base_url: str = OLLAMA_BASE_URL,
    model: str = CLASSIFIER_MODEL
) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """
    Classify a document image using Ollama LLM.
    
    Args:
        image_path: Path to the image file
        base_url: Ollama base URL
        model: Model name for classification
    
    Returns:
        Tuple of (document_type, template_info_dict) or (None, None) if classification fails
    """
    try:
        # Validate image exists
        if not Path(image_path).exists():
            print(f"Error: Image file not found: {image_path}")
            return None, None
        
        # Encode image to base64
        image_b64 = encode_image_to_base64(image_path)
        
        # Prepare the prompt
        document_list = "\n".join([f"- {doc}" for doc in DOCUMENT_NAMES])
        prompt = f"""You are a healthcare document classifier. Analyze this medical document image and identify its type.

Available document types:
{document_list}

Please respond with ONLY the exact document type name from the list above. Do not include any explanation or additional text."""
        
        # Call Ollama API for classification
        url = f"{base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        
        print(f"[1/3] Classifying document with {model}...")
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        classification_text = result.get("response", "").strip()
        
        print(f"      Classification result: {classification_text}")
        
        # Find matching document type
        for doc_type in DOCUMENT_NAMES:
            if doc_type.lower() in classification_text.lower():
                template_info = DOCUMENT_TYPES[doc_type].copy()
                return doc_type, template_info
        
        print(f"Warning: Could not match response to any document type")
        return None, None
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {base_url}")
        print(f"Please ensure Ollama is running on port 11434")
        return None, None
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return None, None


def load_templates(
    template_info: Dict[str, str],
    templates_base_dir: Path
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Load prompt template and JSON template files.
    
    Args:
        template_info: Dictionary with 'prompt_template' and 'template' keys
        templates_base_dir: Base directory for templates (data/prompts/)
    
    Returns:
        Tuple of (prompt_template_text, json_template_dict) or (None, None) if loading fails
    """
    try:
        # Load prompt template
        prompt_file = templates_base_dir / template_info["prompt_template"]
        if not prompt_file.exists():
            print(f"Error: Prompt template not found: {prompt_file}")
            return None, None
        
        prompt_template = prompt_file.read_text()
        
        # Load JSON template
        json_file = templates_base_dir / template_info["template"]
        if not json_file.exists():
            print(f"Error: JSON template not found: {json_file}")
            return None, None
        
        json_template = json.load(open(json_file))
        
        print(f"[2/3] Loaded templates: {template_info['prompt_template']}, {template_info['template']}")
        return prompt_template, json_template
    
    except Exception as e:
        print(f"Error loading templates: {str(e)}")
        return None, None


def extract_information(
    image_path: str,
    prompt_template: str,
    json_template: Dict,
    base_url: str = OLLAMA_BASE_URL,
    model: str = EXTRACTOR_MODEL,
    temperature: float = 0.1,
    top_p: float = 0.8,
    top_k: int = 20
) -> Optional[Dict[str, Any]]:
    """
    Extract structured information from document using Ollama.
    
    Args:
        image_path: Path to the image file
        prompt_template: Prompt template text
        json_template: JSON template dictionary
        base_url: Ollama base URL
        model: Model name for extraction
        temperature: Temperature parameter for generation
        top_p: Top-p parameter
        top_k: Top-k parameter
    
    Returns:
        Extracted information as dictionary or None if extraction fails
    """
    try:
        # Encode image
        image_b64 = encode_image_to_base64(image_path)
        
        # Inject template into prompt
        prompt = prompt_template.replace(
            "{json_template}",
            json.dumps(json_template, ensure_ascii=False, indent=2)
        )
        
        # Call Ollama API for extraction
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64]
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repeat_penalty": 1.0,
                "think": False,
            },
        }
        
        print(f"[3/3] Extracting information with {model}...")
        response = requests.post(url, json=payload, timeout=500)
        response.raise_for_status()
        
        content = response.json()["message"]["content"]
        
        # Parse JSON output
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Strip markdown code fences if present
            lines = content.strip().split("\n")
            if len(lines) > 2 and lines[0].startswith("```"):
                result = json.loads("\n".join(lines[1:-1]))
            else:
                result = json.loads(content)
        
        print(f"      Extraction completed successfully")
        return result
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Ollama at {base_url}")
        return None
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return None


class HealthcareOllamaPipeline:
    """API-style healthcare pipeline with classify and extraction methods."""

    def __init__(
        self,
        templates_base_dir: str = "data/prompts",
        ollama_base_url: str = OLLAMA_BASE_URL,
        classifier_model: str = CLASSIFIER_MODEL,
        extractor_model: str = EXTRACTOR_MODEL,
    ) -> None:
        self.templates_dir = Path(templates_base_dir)
        self.ollama_base_url = ollama_base_url
        self.classifier_model = classifier_model
        self.extractor_model = extractor_model

    def classify(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a single image into a healthcare document type.

        Returns:
            {
                "status": "success" | "error",
                "message": str,
                "document_type": Optional[str],
                "template_info": Optional[Dict[str, str]]
            }
        """
        image = Path(image_path)
        if not image.exists():
            return {
                "status": "error",
                "message": f"Image file not found: {image}",
                "document_type": None,
                "template_info": None,
            }

        if not self.templates_dir.exists():
            return {
                "status": "error",
                "message": f"Templates directory not found: {self.templates_dir}",
                "document_type": None,
                "template_info": None,
            }

        doc_type, template_info = classify_document(
            str(image),
            self.ollama_base_url,
            self.classifier_model,
        )
        if not doc_type:
            return {
                "status": "error",
                "message": "Failed to classify document",
                "document_type": None,
                "template_info": None,
            }

        return {
            "status": "success",
            "message": "Document classified successfully",
            "document_type": doc_type,
            "template_info": template_info,
        }

    def extraction(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract structured information from a single image.

        Execution order:
        1) call classifier
        2) call extraction model
        """
        image = Path(image_path)
        if not image.exists():
            return {
                "status": "error",
                "message": f"Image file not found: {image}",
                "document_type": None,
                "template_filename": None,
                "extracted_data": None,
                "output_path": None,
            }

        if not self.templates_dir.exists():
            return {
                "status": "error",
                "message": f"Templates directory not found: {self.templates_dir}",
                "document_type": None,
                "template_filename": None,
                "extracted_data": None,
                "output_path": None,
            }

        print(f"\n{'='*60}")
        print("Healthcare Document Processing Pipeline")
        print(f"{'='*60}")
        print(f"Input image: {image}")

        # Step 1: call classifier
        classify_result = self.classify(str(image))
        if classify_result["status"] == "error":
            return {
                "status": "error",
                "message": classify_result["message"],
                "document_type": None,
                "template_filename": None,
                "extracted_data": None,
                "output_path": None,
            }

        doc_type = classify_result["document_type"]
        template_info = classify_result["template_info"]
        template_filename = template_info.get("template") if template_info else None

        prompt_template, json_template = load_templates(template_info, self.templates_dir)
        if not prompt_template or not json_template:
            return {
                "status": "error",
                "message": "Failed to load templates",
                "document_type": doc_type,
                "template_filename": template_filename,
                "extracted_data": None,
                "output_path": None,
            }

        # Step 2: call extraction model
        extracted_data = extract_information(
            str(image),
            prompt_template,
            json_template,
            self.ollama_base_url,
            self.extractor_model,
        )
        if not extracted_data:
            return {
                "status": "error",
                "message": "Failed to extract information",
                "document_type": doc_type,
                "template_filename": template_filename,
                "extracted_data": None,
                "output_path": None,
            }

        output_path = None
        if output_dir:
            output_path_obj = Path(output_dir) / image.with_suffix(".json").name
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            output_path_obj.write_text(json.dumps(extracted_data, ensure_ascii=False, indent=2))
            output_path = str(output_path_obj)
            print(f"\n✓ Processing completed successfully")
            print(f"✓ Output saved to: {output_path_obj}\n")

        return {
            "status": "success",
            "document_type": doc_type,
            "template_filename": template_filename,
            "extracted_data": extracted_data,
            "output_path": output_path,
        }


def process_document(
    image_path: str,
    output_dir: str,
    templates_base_dir: str = "data/prompts",
    ollama_base_url: str = OLLAMA_BASE_URL,
    classifier_model: str = CLASSIFIER_MODEL,
    extractor_model: str = EXTRACTOR_MODEL,
) -> Dict[str, Any]:
    """
    Complete pipeline: classify document and extract information.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output JSON
        templates_base_dir: Base directory for prompt/JSON templates
        ollama_base_url: Ollama server URL
        classifier_model: Model for document classification
        extractor_model: Model for information extraction
    
    Returns:
        Dictionary with results or error information
    """
    pipeline = HealthcareOllamaPipeline(
        templates_base_dir=templates_base_dir,
        ollama_base_url=ollama_base_url,
        classifier_model=classifier_model,
        extractor_model=extractor_model,
    )
    return pipeline.extraction(image_path=image_path, output_dir=output_dir)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Healthcare Document Processing Pipeline: Classification + Information Extraction"
    )
    parser.add_argument("--image", required=True, help="Path to input document image")
    parser.add_argument("--output-dir", required=True, help="Directory to save output JSON")
    parser.add_argument(
        "--templates-dir",
        default="data/prompts",
        help="Base directory for prompt and JSON templates (default: data/prompts)"
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_BASE_URL,
        help=f"Ollama server URL (default: {OLLAMA_BASE_URL})"
    )
    parser.add_argument(
        "--classifier-model",
        default=CLASSIFIER_MODEL,
        help=f"Model for document classification (default: {CLASSIFIER_MODEL})"
    )
    parser.add_argument(
        "--extractor-model",
        default=EXTRACTOR_MODEL,
        help=f"Model for information extraction (default: {EXTRACTOR_MODEL})"
    )
    
    args = parser.parse_args()
    
    result = process_document(
        image_path=args.image,
        output_dir=args.output_dir,
        templates_base_dir=args.templates_dir,
        ollama_base_url=args.ollama_url,
        classifier_model=args.classifier_model,
        extractor_model=args.extractor_model,
    )
    
    print("Pipeline Result:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
