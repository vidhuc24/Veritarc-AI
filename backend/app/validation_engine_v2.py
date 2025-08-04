"""
Improved Validation Engine for Veritarc AI (Version 2)
Enhanced LLM-based validation with better prompts, thresholds, and decisiveness
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

class ImprovedValidationEngine:
    def __init__(self):
        """Initialize the improved validation engine"""
        self.openai_client = OpenAI()
        self.retrieval_engine = VeritarcRetrievalEngine()
        
        # Improved scoring thresholds for more decisive classification
        self.scoring_thresholds = {
            "pass_threshold": 0.80,    # >= 0.80 = Pass
            "fail_threshold": 0.60,    # < 0.60 = Fail
            # 0.60 - 0.79 = Partial (narrower range)
        }
        
        # Validation criteria weights (refined)
        self.criteria_weights = {
            "completeness": 0.30,      # Most important
            "accuracy": 0.30,          # Most important
            "approval": 0.25,          # Critical for SOX
            "timeliness": 0.10,        # Less critical
            "documentation": 0.05      # Least critical
        }

    def validate_evidence_against_control(self, evidence_content: str, control_id: str) -> ValidationResult:
        """Validate evidence document against a specific SOX control with improved logic"""
        
        # Get control details
        control_details = self.retrieval_engine.get_control_details(control_id)
        if not control_details:
            raise ValueError(f"Control {control_id} not found")
        
        # Get full control document for validation
        control_docs = self.retrieval_engine.controls_collection.get(ids=[control_id])
        control_content = control_docs['documents'][0] if control_docs['documents'] else ""
        
        # Create improved validation prompt
        validation_prompt = self._create_improved_validation_prompt(evidence_content, control_content, control_details)
        
        # Get LLM validation response
        validation_response = self._get_llm_validation(validation_prompt)
        
        # Parse and improve classification
        parsed_result = self._parse_and_improve_validation_response(validation_response, control_id, evidence_content)
        
        return parsed_result

    def _create_improved_validation_prompt(self, evidence_content: str, control_content: str, control_details: Dict) -> str:
        """Create improved validation prompt with better decisiveness and examples"""
        
        # Extract control category for specific guidance
        category = control_details.get('category', 'Unknown')
        
        # Category-specific guidance
        category_guidance = self._get_category_specific_guidance(category)
        
        prompt = f"""You are a senior SOX compliance auditor with 15+ years of experience. You must make DECISIVE pass/fail decisions based on clear SOX requirements.

CONTROL INFORMATION:
Control ID: {control_details.get('control_id', 'Unknown')}
Control Name: {control_details.get('name', 'Unknown')}
Control Category: {category}

CONTROL REQUIREMENTS:
{control_content}

EVIDENCE DOCUMENT:
{evidence_content[:2500]}

{category_guidance}

VALIDATION TASK - BE DECISIVE:
Evaluate this evidence against SOX control requirements. You MUST classify as either PASS or FAIL. Use PARTIAL only in rare cases where evidence shows mixed results.

SCORING CRITERIA (Rate 0.0-1.0):
1. COMPLETENESS (30%): Are ALL required elements present?
2. ACCURACY (30%): Is information accurate and verifiable?
3. APPROVAL (25%): Are proper approvals and signatures present?
4. TIMELINESS (10%): Is evidence current and within timeframes?
5. DOCUMENTATION (5%): Is formatting professional and complete?

CLASSIFICATION RULES:
- PASS: Score ≥ 0.80 - Evidence clearly meets SOX requirements
- FAIL: Score < 0.60 - Evidence has significant gaps or failures
- PARTIAL: Score 0.60-0.79 - Evidence shows mixed compliance (use sparingly)

EXAMPLES:
✅ PASS Example: "Access review shows all users have manager approval, quarterly certification completed, terminated users removed promptly, documentation complete"
❌ FAIL Example: "Access review missing manager approvals, no termination process, outdated user list, critical gaps in documentation"

Respond in this EXACT JSON format:
{{
    "validation_score": 0.85,
    "confidence_level": "High",
    "assessment": "Clear assessment with specific reasons for pass/fail decision",
    "recommendations": ["Specific actionable recommendation 1", "specific actionable recommendation 2"],
    "gaps_identified": ["Specific gap 1", "Specific gap 2"],
    "compliance_status": "Pass",
    "criteria_scores": {{
        "completeness": 0.9,
        "accuracy": 0.8,
        "approval": 0.9,
        "timeliness": 0.8,
        "documentation": 0.7
    }},
    "key_findings": ["Critical finding 1", "Critical finding 2"],
    "decision_rationale": "Specific reason why this is Pass/Fail with SOX reference"
}}

Be decisive. Auditors need clear guidance, not ambiguous "partial" assessments."""
        
        return prompt

    def _get_category_specific_guidance(self, category: str) -> str:
        """Get category-specific validation guidance"""
        
        guidance_map = {
            "Access Controls": """
ACCESS CONTROL SPECIFIC GUIDANCE:
- PASS requires: Manager approval signatures, current user lists, quarterly reviews, terminated user removal
- FAIL indicators: Missing approvals, outdated access, no termination process, excessive privileges
- Key SOX requirement: Segregation of duties and access management
            """,
            "Change Management": """
CHANGE MANAGEMENT SPECIFIC GUIDANCE:  
- PASS requires: Formal change request, business justification, testing evidence, rollback plan, approvals
- FAIL indicators: No change request, untested changes, missing approvals, no rollback procedures
- Key SOX requirement: Controlled change process with proper authorization
            """,
            "Data Backup & Recovery": """
BACKUP & RECOVERY SPECIFIC GUIDANCE:
- PASS requires: Regular backup schedules, integrity verification, offsite storage, recovery testing
- FAIL indicators: Failed backups, no integrity checks, no recovery testing, missing documentation
- Key SOX requirement: Data protection and business continuity assurance
            """
        }
        
        return guidance_map.get(category, "Apply standard SOX compliance evaluation criteria.")

    def _parse_and_improve_validation_response(self, response: str, control_id: str, evidence_content: str) -> ValidationResult:
        """Parse LLM response and apply improved classification logic"""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                parsed_data = json.loads(response)
            
            # Get raw score and status
            raw_score = parsed_data.get("validation_score", 0.5)
            raw_status = parsed_data.get("compliance_status", "Partial")
            
            # Apply improved classification logic
            improved_score, improved_status = self._apply_improved_classification(raw_score, raw_status, parsed_data)
            
            # Create ValidationResult with improvements
            result = ValidationResult(
                control_id=control_id,
                evidence_id=f"evidence_{hash(evidence_content) % 10000}",
                validation_score=improved_score,
                confidence_level=parsed_data.get("confidence_level", "Medium"),
                assessment=parsed_data.get("assessment", "Assessment not available"),
                recommendations=parsed_data.get("recommendations", []),
                gaps_identified=parsed_data.get("gaps_identified", []),
                compliance_status=improved_status,
                validation_details={
                    "criteria_scores": parsed_data.get("criteria_scores", {}),
                    "key_findings": parsed_data.get("key_findings", []),
                    "decision_rationale": parsed_data.get("decision_rationale", ""),
                    "raw_llm_score": raw_score,
                    "raw_llm_status": raw_status,
                    "classification_method": "improved_thresholds"
                }
            )
            
            return result
            
        except Exception as e:
            print(f"Error parsing validation response: {e}")
            # Return improved default result
            return self._create_improved_fallback_result(control_id, evidence_content)

    def _apply_improved_classification(self, raw_score: float, raw_status: str, parsed_data: Dict) -> tuple[float, str]:
        """Apply improved classification logic with stricter thresholds"""
        
        # Calculate weighted score from criteria if available
        criteria_scores = parsed_data.get("criteria_scores", {})
        if criteria_scores:
            weighted_score = 0.0
            for criterion, weight in self.criteria_weights.items():
                if criterion in criteria_scores:
                    weighted_score += criteria_scores[criterion] * weight
                else:
                    weighted_score += raw_score * weight  # Use raw score as fallback
        else:
            weighted_score = raw_score
        
        # Apply strict thresholds for decisive classification
        if weighted_score >= self.scoring_thresholds["pass_threshold"]:
            final_status = "Pass"
        elif weighted_score < self.scoring_thresholds["fail_threshold"]:
            final_status = "Fail"
        else:
            # Still allow partial, but with stricter criteria
            final_status = "Partial"
        
        # Override partial to pass/fail if evidence is clear
        if final_status == "Partial":
            final_status = self._resolve_partial_classification(weighted_score, parsed_data)
        
        return weighted_score, final_status

    def _resolve_partial_classification(self, score: float, parsed_data: Dict) -> str:
        """Resolve partial classifications to be more decisive"""
        
        # Check for critical compliance indicators
        key_findings = parsed_data.get("key_findings", [])
        gaps_identified = parsed_data.get("gaps_identified", [])
        
        # If significant gaps, lean toward fail
        critical_gaps = ["no approval", "missing", "failed", "expired", "unauthorized", "incomplete"]
        gap_text = " ".join(gaps_identified).lower()
        
        has_critical_gaps = any(gap in gap_text for gap in critical_gaps)
        
        if has_critical_gaps:
            return "Fail"
        elif score >= 0.70:  # Closer to pass threshold
            return "Pass"
        else:
            return "Fail"  # Be stricter - lean toward fail for unclear cases

    def _create_improved_fallback_result(self, control_id: str, evidence_content: str) -> ValidationResult:
        """Create improved fallback result for error cases"""
        
        return ValidationResult(
            control_id=control_id,
            evidence_id=f"evidence_{hash(evidence_content) % 10000}",
            validation_score=0.50,
            confidence_level="Low",
            assessment="Unable to complete validation due to technical issues - requires manual review",
            recommendations=["Conduct manual review", "Verify all required elements", "Ensure proper approvals"],
            gaps_identified=["Automated validation failed"],
            compliance_status="Fail",  # Fail-safe approach
            validation_details={
                "error": "Fallback result due to parsing error",
                "classification_method": "fail_safe_fallback"
            }
        )

    def _get_llm_validation(self, prompt: str) -> str:
        """Get validation response from LLM with improved parameters"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior SOX compliance auditor. Make decisive pass/fail decisions. Avoid ambiguous 'partial' classifications unless truly warranted. Provide structured, objective assessments in the exact JSON format requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent, decisive results
                max_tokens=1200,  # Increased for detailed responses
                presence_penalty=0.1,  # Encourage specific details
                frequency_penalty=0.1   # Reduce repetition
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error getting LLM validation: {e}")
            return self._get_improved_fallback_response()

    def _get_improved_fallback_response(self) -> str:
        """Improved fallback response"""
        return json.dumps({
            "validation_score": 0.50,
            "confidence_level": "Low",
            "assessment": "Technical error occurred during validation - manual review required",
            "recommendations": ["Conduct manual review", "Verify system connectivity", "Re-submit evidence"],
            "gaps_identified": ["Automated validation unavailable"],
            "compliance_status": "Fail",  # Fail-safe
            "criteria_scores": {
                "completeness": 0.5,
                "accuracy": 0.5,
                "approval": 0.5,
                "timeliness": 0.5,
                "documentation": 0.5
            },
            "key_findings": ["Manual review required due to technical issues"],
            "decision_rationale": "Fail-safe classification due to system error"
        })

    def test_improved_validation(self):
        """Test the improved validation engine"""
        print("\n🔧 Testing Improved Validation Engine...")
        
        # Load sample evidence
        try:
            with open('data/synthetic_evidence/test_evidence.json', 'r') as f:
                evidence_docs = json.load(f)
        except FileNotFoundError:
            print("❌ Test evidence file not found")
            return
        
        # Test improved validation on first 3 documents
        print("Testing improved classification on first 3 documents:")
        
        for i in range(3):
            if i < len(evidence_docs):
                evidence_doc = evidence_docs[i]
                control_id = evidence_doc.get('control_id')
                expected = evidence_doc.get('expected_result')
                
                print(f"\nDocument {i+1} ({evidence_doc.get('document_type')}):")
                print(f"  Expected: {expected}")
                
                if control_id:
                    result = self.validate_evidence_against_control(evidence_doc['content'], control_id)
                    print(f"  Predicted: {result.compliance_status}")
                    print(f"  Score: {result.validation_score:.3f}")
                    print(f"  Correct: {self._is_correct(expected, result.compliance_status)}")
                    print(f"  Rationale: {result.validation_details.get('decision_rationale', 'N/A')[:100]}...")

    def _is_correct(self, expected: str, predicted: str) -> bool:
        """Check if prediction is correct"""
        if predicted.lower() == "partial":
            predicted = "pass"  # Map partial to pass for evaluation
        return expected.lower() == predicted.lower()


def main():
    """Test the improved validation engine"""
    print("🚀 Testing Improved Veritarc AI Validation Engine...")
    
    # Initialize improved engine
    engine = ImprovedValidationEngine()
    
    # Test improved functionality
    engine.test_improved_validation()
    
    print("\n✅ Improved Validation Engine Testing Complete!")


if __name__ == "__main__":
    main() 