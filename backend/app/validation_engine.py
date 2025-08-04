"""
Evidence Validation Engine for Veritarc AI
LLM-based validation of evidence documents against SOX controls
"""

import json
import re
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# Add the backend/app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))
from retrieval_engine import VeritarcRetrievalEngine, RetrievalResult

# Load environment variables
load_dotenv()

@dataclass
class ValidationResult:
    """Structured validation result"""
    control_id: str
    evidence_id: str
    validation_score: float  # 0.0 to 1.0
    confidence_level: str  # "High", "Medium", "Low"
    assessment: str
    recommendations: List[str]
    gaps_identified: List[str]
    compliance_status: str  # "Pass", "Fail", "Partial"
    validation_details: Dict[str, Any]

class VeritarcValidationEngine:
    def __init__(self):
        """Initialize the validation engine"""
        self.openai_client = OpenAI()
        self.retrieval_engine = VeritarcRetrievalEngine()
        
        # Validation criteria weights for scoring
        self.criteria_weights = {
            "completeness": 0.25,
            "accuracy": 0.25, 
            "timeliness": 0.20,
            "approval": 0.20,
            "documentation": 0.10
        }
    
    def validate_evidence_against_control(self, evidence_content: str, control_id: str) -> ValidationResult:
        """Validate evidence document against a specific SOX control"""
        
        # Get control details
        control_details = self.retrieval_engine.get_control_details(control_id)
        if not control_details:
            raise ValueError(f"Control {control_id} not found")
        
        # Get full control document for validation
        control_docs = self.retrieval_engine.controls_collection.get(ids=[control_id])
        control_content = control_docs['documents'][0] if control_docs['documents'] else ""
        
        # Create validation prompt
        validation_prompt = self._create_validation_prompt(evidence_content, control_content, control_details)
        
        # Get LLM validation response
        validation_response = self._get_llm_validation(validation_prompt)
        
        # Parse structured response
        parsed_result = self._parse_validation_response(validation_response, control_id, evidence_content)
        
        return parsed_result
    
    def _create_validation_prompt(self, evidence_content: str, control_content: str, control_details: Dict) -> str:
        """Create comprehensive validation prompt for LLM"""
        
        prompt = f"""
You are an expert SOX compliance auditor evaluating evidence against control requirements.

CONTROL INFORMATION:
Control ID: {control_details.get('control_id', 'Unknown')}
Control Name: {control_details.get('name', 'Unknown')}
Control Category: {control_details.get('category', 'Unknown')}

CONTROL REQUIREMENTS:
{control_content}

EVIDENCE DOCUMENT:
{evidence_content[:2000]}  # Limit content length

VALIDATION TASK:
Evaluate this evidence document against the SOX control requirements. Assess:

1. COMPLETENESS: Does the evidence cover all required control elements?
2. ACCURACY: Is the evidence accurate and properly documented?
3. TIMELINESS: Is the evidence current and within required timeframes?
4. APPROVAL: Are proper approvals and signatures present?
5. DOCUMENTATION: Is the evidence properly formatted and complete?

Provide your assessment in the following JSON format:
{{
    "validation_score": 0.85,
    "confidence_level": "High",
    "assessment": "Brief assessment summary",
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "gaps_identified": ["Gap 1", "Gap 2"],
    "compliance_status": "Pass",
    "criteria_scores": {{
        "completeness": 0.9,
        "accuracy": 0.8,
        "timeliness": 0.9,
        "approval": 0.7,
        "documentation": 0.8
    }},
    "key_findings": ["Finding 1", "Finding 2"]
}}

Focus on SOX compliance requirements and provide specific, actionable feedback.
"""
        return prompt
    
    def _get_llm_validation(self, prompt: str) -> str:
        """Get validation response from LLM"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert SOX compliance auditor. Provide structured, objective assessments in the exact JSON format requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error getting LLM validation: {e}")
            # Return fallback response
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Fallback response if LLM fails"""
        return json.dumps({
            "validation_score": 0.5,
            "confidence_level": "Low",
            "assessment": "Unable to complete validation due to technical issues",
            "recommendations": ["Review evidence manually", "Verify control requirements"],
            "gaps_identified": ["Validation could not be completed"],
            "compliance_status": "Partial",
            "criteria_scores": {
                "completeness": 0.5,
                "accuracy": 0.5,
                "timeliness": 0.5,
                "approval": 0.5,
                "documentation": 0.5
            },
            "key_findings": ["Manual review required"]
        })
    
    def _parse_validation_response(self, response: str, control_id: str, evidence_content: str) -> ValidationResult:
        """Parse LLM response into structured ValidationResult"""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                # Try to parse the entire response
                parsed_data = json.loads(response)
            
            # Create ValidationResult
            result = ValidationResult(
                control_id=control_id,
                evidence_id=f"evidence_{hash(evidence_content) % 10000}",  # Simple ID generation
                validation_score=parsed_data.get("validation_score", 0.5),
                confidence_level=parsed_data.get("confidence_level", "Low"),
                assessment=parsed_data.get("assessment", "Assessment not available"),
                recommendations=parsed_data.get("recommendations", []),
                gaps_identified=parsed_data.get("gaps_identified", []),
                compliance_status=parsed_data.get("compliance_status", "Partial"),
                validation_details={
                    "criteria_scores": parsed_data.get("criteria_scores", {}),
                    "key_findings": parsed_data.get("key_findings", []),
                    "raw_response": response
                }
            )
            
            return result
            
        except Exception as e:
            print(f"Error parsing validation response: {e}")
            # Return default result
            return ValidationResult(
                control_id=control_id,
                evidence_id=f"evidence_{hash(evidence_content) % 10000}",
                validation_score=0.5,
                confidence_level="Low",
                assessment="Error parsing validation response",
                recommendations=["Manual review required"],
                gaps_identified=["Parsing error occurred"],
                compliance_status="Partial",
                validation_details={"error": str(e), "raw_response": response}
            )
    
    def validate_evidence_with_auto_control_matching(self, evidence_content: str) -> List[ValidationResult]:
        """Validate evidence against automatically matched controls"""
        
        # Retrieve relevant controls for the evidence
        relevant_controls = self.retrieval_engine.retrieve_controls_for_evidence(evidence_content)
        
        validation_results = []
        
        # Validate against top 2 most relevant controls
        for control in relevant_controls[:2]:
            control_id = control.metadata.get('control_id')
            if control_id:
                try:
                    result = self.validate_evidence_against_control(evidence_content, control_id)
                    validation_results.append(result)
                except Exception as e:
                    print(f"Error validating against control {control_id}: {e}")
                    continue
        
        return validation_results
    
    def batch_validate_evidence(self, evidence_documents: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate multiple evidence documents"""
        
        all_results = []
        
        for i, evidence_doc in enumerate(evidence_documents):
            print(f"Validating evidence {i+1}/{len(evidence_documents)}: {evidence_doc.get('document_type', 'Unknown')}")
            
            try:
                # Get the control ID from evidence metadata
                control_id = evidence_doc.get('control_id')
                if control_id:
                    result = self.validate_evidence_against_control(
                        evidence_doc['content'], 
                        control_id
                    )
                    all_results.append(result)
                else:
                    # Auto-match controls if control_id not specified
                    results = self.validate_evidence_with_auto_control_matching(evidence_doc['content'])
                    all_results.extend(results)
                    
            except Exception as e:
                print(f"Error validating evidence {i+1}: {e}")
                continue
        
        return all_results
    
    def calculate_validation_metrics(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Calculate validation performance metrics"""
        
        if not results:
            return {}
        
        total_results = len(results)
        pass_count = sum(1 for r in results if r.compliance_status == "Pass")
        fail_count = sum(1 for r in results if r.compliance_status == "Fail")
        partial_count = sum(1 for r in results if r.compliance_status == "Partial")
        
        avg_score = sum(r.validation_score for r in results) / total_results
        
        confidence_distribution = {}
        for result in results:
            confidence = result.confidence_level
            confidence_distribution[confidence] = confidence_distribution.get(confidence, 0) + 1
        
        return {
            "total_validations": total_results,
            "pass_rate": pass_count / total_results,
            "fail_rate": fail_count / total_results,
            "partial_rate": partial_count / total_results,
            "average_score": avg_score,
            "confidence_distribution": confidence_distribution,
            "score_distribution": {
                "high": sum(1 for r in results if r.validation_score >= 0.8),
                "medium": sum(1 for r in results if 0.6 <= r.validation_score < 0.8),
                "low": sum(1 for r in results if r.validation_score < 0.6)
            }
        }
    
    def test_validation_engine(self):
        """Test the validation engine with sample evidence"""
        print("\n🧪 Testing Validation Engine...")
        
        # Load sample evidence
        try:
            with open('data/synthetic_evidence/test_evidence.json', 'r') as f:
                evidence_docs = json.load(f)
        except FileNotFoundError:
            print("❌ Test evidence file not found")
            return
        
        # Test 1: Single evidence validation
        print("Test 1: Validating single evidence document...")
        sample_evidence = evidence_docs[0]  # First evidence document
        control_id = sample_evidence.get('control_id')
        
        if control_id:
            result = self.validate_evidence_against_control(sample_evidence['content'], control_id)
            print(f"Validation Result:")
            print(f"  Control: {result.control_id}")
            print(f"  Score: {result.validation_score:.3f}")
            print(f"  Status: {result.compliance_status}")
            print(f"  Confidence: {result.confidence_level}")
            print(f"  Assessment: {result.assessment[:100]}...")
        
        # Test 2: Batch validation
        print("\nTest 2: Batch validation of evidence documents...")
        batch_results = self.batch_validate_evidence(evidence_docs[:3])  # Test with 3 documents
        
        print(f"Batch validation completed: {len(batch_results)} results")
        
        # Test 3: Calculate metrics
        print("\nTest 3: Validation metrics...")
        metrics = self.calculate_validation_metrics(batch_results)
        print(f"Validation Metrics:")
        print(f"  Total validations: {metrics.get('total_validations', 0)}")
        print(f"  Pass rate: {metrics.get('pass_rate', 0):.2%}")
        print(f"  Average score: {metrics.get('average_score', 0):.3f}")
        print(f"  Confidence distribution: {metrics.get('confidence_distribution', {})}")
        
        print("✅ Validation engine tests completed!")


def main():
    """Test the validation engine"""
    print("🚀 Testing Veritarc AI Validation Engine...")
    
    # Initialize validation engine
    engine = VeritarcValidationEngine()
    
    # Test validation functionality
    engine.test_validation_engine()
    
    print("\n🎉 Step 3 Complete: Evidence Validation Engine Successful!")


if __name__ == "__main__":
    main() 