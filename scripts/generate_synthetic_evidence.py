"""
Synthetic Evidence Document Generator
Uses GPT-4 to create realistic audit evidence documents for testing
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class SyntheticEvidenceGenerator:
    def __init__(self):
        self.client = OpenAI()
        self.companies = [
            {"name": "RideShare Pro", "type": "ride-sharing", "employees": 2500},
            {"name": "PayFlow Tech", "type": "fintech", "employees": 800},
            {"name": "MarketPlace Inc", "type": "e-commerce", "employees": 1200},
            {"name": "FinanceFirst", "type": "fintech", "employees": 600},
            {"name": "DeliveryNow", "type": "ride-sharing", "employees": 1800}
        ]
        
        # Load SOX controls
        with open('data/sample_sox_controls.json', 'r') as f:
            self.sox_controls = json.load(f)
    
    def generate_access_review_report(self, quality_level: str, company: Dict) -> Dict[str, Any]:
        """Generate synthetic access review report"""
        
        quality_prompts = {
            "high": "Create a comprehensive, well-documented access review report with all required approvals, signatures, and complete information.",
            "medium": "Create an access review report with most information present but minor gaps like missing signatures or incomplete dates.",
            "low": "Create an access review report with significant gaps like missing approvals, outdated information, or incomplete user lists.",
            "fail": "Create an access review report with critical failures like no management approval, expired reviews, or unauthorized access."
        }
        
        prompt = f"""
        Create a realistic access review report for {company['name']}, a {company['type']} company with {company['employees']} employees.
        
        Quality Level: {quality_level}
        Instructions: {quality_prompts[quality_level]}
        
        The report should include:
        - Company header and report date
        - Review period (quarterly)
        - List of 8-12 users with roles, access levels, last login dates
        - Manager approval status
        - Findings and recommendations
        - Review completion status
        
        Make it look like a real audit document with realistic employee names, roles relevant to {company['type']} business, and specific system access details.
        
        Format as a structured text document that could be converted to PDF.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in creating realistic audit evidence documents. Generate professional, authentic-looking evidence that auditors would typically review."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return {
            "document_type": "access_review_report",
            "control_id": "SOX-ITGC-AC-02",
            "company": company["name"],
            "quality_level": quality_level,
            "content": response.choices[0].message.content,
            "expected_result": "pass" if quality_level in ["high", "medium"] else "fail",
            "generated_date": datetime.now().isoformat()
        }
    
    def generate_change_request_form(self, quality_level: str, company: Dict) -> Dict[str, Any]:
        """Generate synthetic change request form"""
        
        quality_prompts = {
            "high": "Create a complete change request with all required approvals, detailed testing plans, rollback procedures, and proper documentation.",
            "medium": "Create a change request with most required elements but minor issues like incomplete testing documentation or missing secondary approvals.",
            "low": "Create a change request with significant gaps like missing business justification, incomplete approvals, or no rollback plan.",
            "fail": "Create a change request with critical failures like no approvals, no testing, or emergency changes without proper justification."
        }
        
        prompt = f"""
        Create a realistic system change request form for {company['name']}, a {company['type']} company.
        
        Quality Level: {quality_level}
        Instructions: {quality_prompts[quality_level]}
        
        The change request should include:
        - Change request ID and date
        - System/application being changed (relevant to {company['type']} business)
        - Business justification
        - Technical details of the change
        - Impact assessment
        - Testing plan and results
        - Approval signatures (technical and business)
        - Implementation date and rollback procedures
        
        Make it realistic with appropriate technical details for a {company['type']} company's financial systems.
        
        Format as a structured form document.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in creating realistic IT change management documentation. Generate professional change request forms that would be used in real organizations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return {
            "document_type": "change_request_form",
            "control_id": "SOX-ITGC-CM-01", 
            "company": company["name"],
            "quality_level": quality_level,
            "content": response.choices[0].message.content,
            "expected_result": "pass" if quality_level in ["high", "medium"] else "fail",
            "generated_date": datetime.now().isoformat()
        }
    
    def generate_backup_verification_log(self, quality_level: str, company: Dict) -> Dict[str, Any]:
        """Generate synthetic backup verification log"""
        
        quality_prompts = {
            "high": "Create a comprehensive backup verification log with all backups successful, integrity checks passed, and complete documentation.",
            "medium": "Create a backup log with most backups successful but minor issues like occasional warnings or incomplete logs.",  
            "low": "Create a backup log with significant issues like failed backups, missing integrity checks, or incomplete documentation.",
            "fail": "Create a backup log with critical failures like multiple backup failures, no verification, or missing critical system backups."
        }
        
        prompt = f"""
        Create a realistic backup verification log for {company['name']}, a {company['type']} company.
        
        Quality Level: {quality_level}
        Instructions: {quality_prompts[quality_level]}
        
        The backup log should include:
        - Report header with company name and date range
        - Daily backup status for critical systems (payment, customer data, financial reporting)
        - Backup completion times and file sizes
        - Integrity check results
        - Failed backup notifications (if any)
        - Offsite backup transfer status
        - Recovery testing results
        - System administrator review and sign-off
        
        Include realistic technical details for {company['type']} business systems and databases.
        
        Format as a technical log report.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in creating realistic IT backup and recovery documentation. Generate professional backup logs that system administrators would actually produce."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return {
            "document_type": "backup_verification_log",
            "control_id": "SOX-ITGC-DR-01",
            "company": company["name"], 
            "quality_level": quality_level,
            "content": response.choices[0].message.content,
            "expected_result": "pass" if quality_level in ["high", "medium"] else "fail",
            "generated_date": datetime.now().isoformat()
        }
    
    def generate_evidence_batch(self, num_documents: int = 20) -> List[Dict[str, Any]]:
        """Generate a batch of evidence documents across all types and quality levels"""
        
        evidence_documents = []
        quality_levels = ["high", "medium", "low", "fail"]
        
        # Distribute documents across types and quality levels
        document_types = [
            self.generate_access_review_report,
            self.generate_change_request_form, 
            self.generate_backup_verification_log
        ]
        
        for i in range(num_documents):
            # Rotate through document types and quality levels
            doc_type_func = document_types[i % len(document_types)]
            quality = quality_levels[i % len(quality_levels)]
            company = random.choice(self.companies)
            
            print(f"Generating document {i+1}/{num_documents}: {doc_type_func.__name__} ({quality} quality)")
            
            try:
                document = doc_type_func(quality, company)
                evidence_documents.append(document)
            except Exception as e:
                print(f"Error generating document {i+1}: {e}")
                continue
                
        return evidence_documents
    
    def save_evidence_documents(self, documents: List[Dict[str, Any]], filename: str = "synthetic_evidence.json"):
        """Save generated evidence documents to file"""
        
        output_path = f"data/synthetic_evidence/{filename}"
        os.makedirs("data/synthetic_evidence", exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(documents, f, indent=2)
        
        print(f"Saved {len(documents)} evidence documents to {output_path}")
        
        # Generate summary statistics
        summary = {
            "total_documents": len(documents),
            "by_type": {},
            "by_quality": {},
            "by_expected_result": {}
        }
        
        for doc in documents:
            # Count by type
            doc_type = doc["document_type"]
            summary["by_type"][doc_type] = summary["by_type"].get(doc_type, 0) + 1
            
            # Count by quality
            quality = doc["quality_level"]
            summary["by_quality"][quality] = summary["by_quality"].get(quality, 0) + 1
            
            # Count by expected result
            result = doc["expected_result"]
            summary["by_expected_result"][result] = summary["by_expected_result"].get(result, 0) + 1
        
        # Save summary
        with open(f"data/synthetic_evidence/summary_{filename}", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Generation Summary:")
        print(f"  Total documents: {summary['total_documents']}")
        print(f"  By type: {summary['by_type']}")
        print(f"  By quality: {summary['by_quality']}")
        print(f"  By expected result: {summary['by_expected_result']}")


if __name__ == "__main__":
    generator = SyntheticEvidenceGenerator()
    
    # Generate a small test batch first
    print("Generating test batch of synthetic evidence documents...")
    test_documents = generator.generate_evidence_batch(9)  # 3 of each type
    generator.save_evidence_documents(test_documents, "test_evidence.json") 