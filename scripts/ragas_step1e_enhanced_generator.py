"""
Step 1e: Enhanced Synthetic Evidence Generator
Incorporates RAGAS insights to create more realistic and sophisticated evidence
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGASEnhancedEvidenceGenerator:
    def __init__(self):
        """Initialize enhanced evidence generator with RAGAS insights"""
        self.client = OpenAI()
        
        # Enhanced company profiles with more detail
        self.companies = [
            {
                "name": "RideShare Pro", 
                "type": "ride-sharing", 
                "employees": 2500,
                "it_team_size": 45,
                "ciso": "Michael Chen",
                "it_director": "Sarah Rodriguez"
            },
            {
                "name": "PayFlow Tech", 
                "type": "fintech", 
                "employees": 800,
                "it_team_size": 25,
                "ciso": "David Kim",
                "it_director": "Lisa Thompson"
            },
            {
                "name": "MarketPlace Inc", 
                "type": "e-commerce", 
                "employees": 1200,
                "it_team_size": 35,
                "ciso": "Jennifer Wu",
                "it_director": "Robert Martinez"
            }
        ]
        
        # Load SOX controls for reference
        with open('data/sample_sox_controls.json', 'r') as f:
            self.sox_controls = json.load(f)
        
        # RAGAS-informed role definitions
        self.roles = {
            "access_controls": [
                {"title": "IT Security Analyst", "name": "Alex Johnson"},
                {"title": "Identity & Access Manager", "name": "Maria Santos"},
                {"title": "Compliance Officer", "name": "Tom Wilson"},
                {"title": "Department Manager", "name": "Emily Davis"}
            ],
            "change_management": [
                {"title": "DevOps Engineer", "name": "Ryan Park"},
                {"title": "Release Manager", "name": "Jessica Lee"},
                {"title": "QA Lead", "name": "Carlos Mendez"},
                {"title": "Technical Director", "name": "Angela Foster"}
            ],
            "backup_recovery": [
                {"title": "Infrastructure Engineer", "name": "Kevin Brown"},
                {"title": "Backup Administrator", "name": "Nicole Garcia"},
                {"title": "Data Protection Officer", "name": "James Liu"},
                {"title": "Operations Manager", "name": "Rachel Green"}
            ]
        }

    def generate_enhanced_access_review_report(self, quality_level: str, company: Dict) -> Dict[str, Any]:
        """Generate enhanced access review report with RAGAS insights"""
        
        # Get relevant SOX control
        access_control = next(c for c in self.sox_controls if c['control_id'] == 'SOX-ITGC-AC-01')
        
        # RAGAS Insight: Reference specific control IDs and validation criteria
        control_reference = f"Control Reference: {access_control['control_id']} - {access_control['name']}"
        validation_criteria = access_control['validation_criteria']
        
        # RAGAS Insight: Include role-specific details
        roles = self.roles["access_controls"]
        reviewer = roles[0]  # IT Security Analyst
        approver = roles[3]  # Department Manager
        
        # RAGAS Insight: Add workflow details
        workflow_steps = [
            "1. Quarterly access review initiated by IT Security",
            "2. Department managers review user access lists", 
            "3. Terminated users identified and flagged",
            "4. Manager approval signatures collected",
            "5. Final CISO certification completed"
        ]
        
        # Enhanced quality-specific prompts with RAGAS insights
        quality_prompts = {
            "high": f"""Create a comprehensive access review report that EXPLICITLY addresses these SOX validation criteria:
{chr(10).join(f"✓ {criteria}" for criteria in validation_criteria)}

Include these workflow details:
{chr(10).join(workflow_steps)}

Role assignments:
- Reviewer: {reviewer['name']} ({reviewer['title']})
- Approver: {approver['name']} ({approver['title']})
- CISO: {company['ciso']}

Show COMPLETE compliance with all criteria.""",
            
            "medium": f"""Create an access review report that addresses most SOX validation criteria but has minor gaps:
{chr(10).join(f"• {criteria}" for criteria in validation_criteria[:4])}

Include workflow but with some missing steps:
{chr(10).join(workflow_steps[:3])}

Role assignments:
- Reviewer: {reviewer['name']} ({reviewer['title']})
- Approver: {approver['name']} ({approver['title']})

Missing: Final CISO signature or 1-2 terminated users not processed.""",
            
            "low": f"""Create an access review report with significant gaps in SOX compliance:
Only addresses: {validation_criteria[0]} and {validation_criteria[2]}

Incomplete workflow:
{workflow_steps[0]}
{workflow_steps[1]}

Role issues:
- Reviewer: {reviewer['name']} (but missing credentials)
- No proper approver signature

Multiple terminated users still have active access.""",
            
            "fail": f"""Create an access review report that FAILS SOX control {access_control['control_id']}:

CRITICAL FAILURES:
- No management approval signatures
- Terminated users still have full system access
- No multi-factor authentication verification
- Review is 6 months overdue
- Missing segregation of duties validation

Workflow completely broken - manual process with no oversight."""
        }
        
        prompt = f"""
Create a realistic quarterly access review report for {company['name']}, a {company['type']} company.
{control_reference}

Quality Level: {quality_level}
Enhanced Requirements: {quality_prompts[quality_level]}

The report must include:
- Company header with {company['name']} branding
- Review period: Q4 2024 (Oct-Dec)
- User access matrix for {company['it_team_size']} IT users + 50 business users
- Specific system access (financial systems, payment processing, customer data)
- Role-based access controls (RBAC) compliance
- Multi-factor authentication status
- Terminated employee access status
- Manager approval signatures with dates
- Findings and recommendations section

Make it look like a real corporate audit document with specific employee names, 
system names relevant to {company['type']} business, and realistic timestamps.
Format as a structured document suitable for SOX audit review.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert SOX compliance auditor creating realistic access review reports with specific control references and detailed workflow documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower for more consistent results
            max_tokens=2000   # Increased for detailed output
        )
        
        return {
            "document_type": "access_review_report",
            "control_id": "SOX-ITGC-AC-01",
            "company": company["name"],
            "quality_level": quality_level,
            "content": response.choices[0].message.content,
            "expected_result": "pass" if quality_level in ["high", "medium"] else "fail",
            "generated_date": datetime.now().isoformat(),
            "enhancements_applied": [
                "specific_control_reference",
                "validation_criteria_explicit",
                "role_specific_details", 
                "workflow_documentation",
                "enhanced_quality_variations"
            ]
        }

    def generate_enhanced_change_request_form(self, quality_level: str, company: Dict) -> Dict[str, Any]:
        """Generate enhanced change request form with RAGAS insights"""
        
        # Get relevant SOX control
        change_control = next(c for c in self.sox_controls if c['control_id'] == 'SOX-ITGC-CM-01')
        
        # RAGAS Insight: Include explicit validation criteria testing
        validation_criteria = change_control['validation_criteria']
        
        # RAGAS Insight: Role-specific details and workflow
        roles = self.roles["change_management"]
        requester = roles[0]  # DevOps Engineer
        approver = roles[3]   # Technical Director
        
        workflow_steps = [
            "1. Change request submitted with business justification",
            "2. Technical impact assessment completed", 
            "3. Security review and approval obtained",
            "4. Testing plan developed and executed",
            "5. Rollback procedures documented",
            "6. Final approval and implementation scheduled"
        ]
        
        quality_prompts = {
            "high": f"""Create a comprehensive change request form that demonstrates FULL compliance with {change_control['control_id']}:

VALIDATION CRITERIA TESTING:
{chr(10).join(f"✅ {criteria}" for criteria in validation_criteria)}

COMPLETE WORKFLOW:
{chr(10).join(workflow_steps)}

Role Details:
- Requester: {requester['name']} ({requester['title']})
- Approver: {approver['name']} ({approver['title']})
- Technical Director: {company['it_director']}

All documentation complete, testing passed, rollback ready.""",
            
            "medium": f"""Create a change request with minor compliance gaps:

MOSTLY ADDRESSED CRITERIA:
{chr(10).join(f"• {criteria}" for criteria in validation_criteria[:4])}

PARTIAL WORKFLOW:
{chr(10).join(workflow_steps[:4])}

Role Details:
- Requester: {requester['name']} ({requester['title']})
- Approver: {approver['name']} (signature pending)

Missing: Complete rollback testing or final security sign-off.""",
            
            "fail": f"""Create a change request that FAILS {change_control['control_id']} requirements:

CRITICAL FAILURES:
- No proper business justification
- Untested changes pushed to production
- No rollback procedures documented
- Missing security approval
- Emergency deployment without proper authorization

Workflow bypassed - direct production deployment."""
        }
        
        prompt = f"""
Create a software change request form for {company['name']}.
Control Reference: {change_control['control_id']} - {change_control['name']}

Quality Level: {quality_level}
Requirements: {quality_prompts[quality_level]}

The form should include:
- Change request ID: CHG-2024-{1000 + hash(company['name']) % 100}
- Business justification for the change
- Technical details and impact assessment  
- Testing plan and results
- Security review status
- Rollback procedures
- Approval signatures and timestamps
- Implementation schedule

Context: Updating payment processing system for {company['type']} operations.
Make it realistic with specific technical details, system names, and proper change management workflow.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior change management specialist creating detailed change request documentation with explicit SOX compliance validation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return {
            "document_type": "change_request_form",
            "control_id": "SOX-ITGC-CM-01", 
            "company": company["name"],
            "quality_level": quality_level,
            "content": response.choices[0].message.content,
            "expected_result": "pass" if quality_level in ["high", "medium"] else "fail",
            "generated_date": datetime.now().isoformat(),
            "enhancements_applied": [
                "explicit_validation_criteria",
                "detailed_workflow_process",
                "role_specific_assignments",
                "control_id_reference"
            ]
        }

    def generate_enhanced_backup_verification_log(self, quality_level: str, company: Dict) -> Dict[str, Any]:
        """Generate enhanced backup verification log with RAGAS insights"""
        
        # Get relevant SOX control
        backup_control = next(c for c in self.sox_controls if c['control_id'] == 'SOX-ITGC-DR-01')
        
        # RAGAS Insight: Process workflow details and role assignments
        roles = self.roles["backup_recovery"]
        operator = roles[1]  # Backup Administrator
        verifier = roles[3]  # Operations Manager
        
        workflow_process = [
            "1. Automated backup job initiated at scheduled time",
            "2. Backup completion status verified",
            "3. Data integrity checks performed",
            "4. Offsite replication confirmed", 
            "5. Recovery testing executed monthly",
            "6. Verification logs reviewed and approved"
        ]
        
        validation_criteria = backup_control['validation_criteria']
        
        quality_prompts = {
            "high": f"""Create a comprehensive backup verification log showing FULL compliance with {backup_control['control_id']}:

PROCESS WORKFLOW DEMONSTRATED:
{chr(10).join(workflow_process)}

VALIDATION CRITERIA VERIFIED:
{chr(10).join(f"✅ {criteria}" for criteria in validation_criteria)}

Role Assignments:  
- Backup Operator: {operator['name']} ({operator['title']})
- Verifier: {verifier['name']} ({verifier['title']})
- Data Protection Officer: {roles[2]['name']}

All backups successful, integrity verified, recovery tested.""",
            
            "fail": f"""Create a backup verification log that FAILS {backup_control['control_id']}:

CRITICAL FAILURES:
- Multiple backup jobs failed without resolution
- No integrity verification performed
- Recovery testing not conducted in 6+ months
- Offsite storage not confirmed
- Missing operator verification signatures

Process breakdown - manual intervention required."""
        }
        
        prompt = f"""
Create a daily backup verification log for {company['name']}.
Control Reference: {backup_control['control_id']} - {backup_control['name']}

Quality Level: {quality_level}
Requirements: {quality_prompts[quality_level]}

The log should include:
- Date range: Past 7 days of backup operations
- System coverage: Financial systems, customer databases, application servers
- Backup job status (success/failure) with timestamps
- Data integrity verification results
- Offsite replication confirmation
- Recovery testing status (monthly requirement)
- Operator verification signatures
- Issues and remediation actions

Make it look like real backup monitoring output with specific system names, 
file sizes, completion times, and technical details for a {company['type']} company.
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an experienced backup administrator creating detailed verification logs with comprehensive process workflow documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return {
            "document_type": "backup_verification_log",
            "control_id": "SOX-ITGC-DR-01",
            "company": company["name"], 
            "quality_level": quality_level,
            "content": response.choices[0].message.content,
            "expected_result": "pass" if quality_level in ["high", "medium"] else "fail",
            "generated_date": datetime.now().isoformat(),
            "enhancements_applied": [
                "process_workflow_details",
                "role_specific_operators",
                "control_validation_explicit",
                "technical_detail_enhancement"
            ]
        }

    def generate_enhanced_evidence_set(self) -> List[Dict[str, Any]]:
        """Generate complete enhanced evidence set"""
        
        print("🚀 Generating Enhanced Evidence Set with RAGAS Insights...")
        
        evidence_documents = []
        
        # Generate evidence for each company and quality level
        for i, company in enumerate(self.companies):
            
            print(f"📋 Generating evidence for {company['name']}...")
            
            # Access Review Report (high quality)
            if i == 0:
                evidence_documents.append(
                    self.generate_enhanced_access_review_report("high", company)
                )
            # Change Request Form (medium quality) 
            elif i == 1:
                evidence_documents.append(
                    self.generate_enhanced_change_request_form("medium", company)
                )
            # Backup Verification Log (fail quality)
            else:
                evidence_documents.append(
                    self.generate_enhanced_backup_verification_log("fail", company)
                )
        
        print(f"✅ Generated {len(evidence_documents)} enhanced evidence documents")
        
        # Save enhanced evidence
        os.makedirs('data/enhanced_evidence', exist_ok=True)
        with open('data/enhanced_evidence/ragas_enhanced_evidence.json', 'w') as f:
            json.dump(evidence_documents, f, indent=2)
        
        # Create summary
        summary = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_documents": len(evidence_documents),
            "enhancements_applied": [
                "RAGAS_control_id_references",
                "explicit_validation_criteria",
                "role_specific_details",
                "workflow_process_documentation", 
                "enhanced_quality_variations"
            ],
            "document_distribution": {
                "access_review_reports": len([d for d in evidence_documents if d['document_type'] == 'access_review_report']),
                "change_request_forms": len([d for d in evidence_documents if d['document_type'] == 'change_request_form']),
                "backup_verification_logs": len([d for d in evidence_documents if d['document_type'] == 'backup_verification_log'])
            },
            "quality_distribution": {
                "high": len([d for d in evidence_documents if d['quality_level'] == 'high']),
                "medium": len([d for d in evidence_documents if d['quality_level'] == 'medium']), 
                "low": len([d for d in evidence_documents if d['quality_level'] == 'low']),
                "fail": len([d for d in evidence_documents if d['quality_level'] == 'fail'])
            }
        }
        
        with open('data/enhanced_evidence/enhancement_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Enhancement Summary:")
        print(f"  Total documents: {summary['total_documents']}")
        print(f"  Enhancements applied: {len(summary['enhancements_applied'])}")
        print(f"  Quality distribution: {summary['quality_distribution']}")
        
        return evidence_documents


def test_enhanced_evidence_generation():
    """Test the enhanced evidence generator"""
    
    print("🚀 Testing RAGAS-Enhanced Evidence Generation")
    print("=" * 60)
    
    try:
        # Initialize enhanced generator
        generator = RAGASEnhancedEvidenceGenerator()
        
        # Generate enhanced evidence set
        enhanced_evidence = generator.generate_enhanced_evidence_set()
        
        # Test analysis
        print(f"\n📋 Enhanced Evidence Analysis:")
        
        for i, doc in enumerate(enhanced_evidence):
            print(f"\n  Document {i+1}: {doc['document_type']}")
            print(f"    Company: {doc['company']}")
            print(f"    Control: {doc['control_id']}")
            print(f"    Quality: {doc['quality_level']}")
            print(f"    Expected: {doc['expected_result']}")
            print(f"    Enhancements: {len(doc['enhancements_applied'])}")
            print(f"    Content length: {len(doc['content'])} chars")
            print(f"    Preview: {doc['content'][:150]}...")
        
        print(f"\n✅ Step 1e Test Results:")
        print(f"  ✓ Enhanced evidence generator created successfully")
        print(f"  ✓ Generated {len(enhanced_evidence)} enhanced documents")
        print(f"  ✓ All documents include RAGAS insights")
        print(f"  ✓ Enhanced evidence saved to data/enhanced_evidence/")
        
        print(f"\n🎯 Step 1e: COMPLETE - Enhanced Evidence Generation Successful!")
        
        return enhanced_evidence
        
    except Exception as e:
        print(f"❌ Error in Step 1e: {e}")
        raise


if __name__ == "__main__":
    test_enhanced_evidence_generation() 