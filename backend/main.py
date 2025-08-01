"""
AuditFlow AI - Main FastAPI Application
AI-Powered Control Evidence Validation System
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AuditFlow AI API",
    description="AI-Powered Control Evidence Validation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ValidationRequest(BaseModel):
    control_id: str
    framework: str
    confidence_threshold: Optional[float] = 0.7

class ValidationResponse(BaseModel):
    validation_score: float
    confidence_level: str
    assessment: str
    recommendations: list[str]
    gaps_identified: list[str]
    compliance_status: str

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "🔍 AuditFlow AI - Evidence Validator API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "auditflow-ai"}

# Main evidence validation endpoint
@app.post("/validate-evidence", response_model=ValidationResponse)
async def validate_evidence(
    file: UploadFile = File(...),
    control_id: str = "SOX-ITGC-01",
    framework: str = "SOX"
):
    """
    Upload and validate evidence document against control requirements
    
    Args:
        file: Evidence document (PDF, DOCX, PNG, JPG)
        control_id: Specific control identifier (e.g., SOX-ITGC-01)
        framework: Compliance framework (SOX, SOC2, ISO27001)
    
    Returns:
        ValidationResponse with assessment and recommendations
    """
    
    # Validate file type
    allowed_extensions = {'pdf', 'docx', 'png', 'jpg', 'jpeg'}
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type '{file_extension}' not supported. Allowed: {allowed_extensions}"
        )
    
    # TODO: Implement actual validation logic
    # For now, return a mock response
    return ValidationResponse(
        validation_score=0.85,
        confidence_level="High",
        assessment="Evidence document appears to meet most control requirements. Minor gaps identified in approval documentation.",
        recommendations=[
            "Include manager approval signature",
            "Add timestamp for change implementation",
            "Provide rollback procedure documentation"
        ],
        gaps_identified=[
            "Missing approval signature",
            "Incomplete timestamp information"
        ],
        compliance_status="Partially Compliant"
    )

# Get available control frameworks
@app.get("/frameworks")
async def get_frameworks():
    return {
        "frameworks": [
            {"id": "SOX", "name": "Sarbanes-Oxley Act", "controls": 15},
            {"id": "SOC2", "name": "SOC 2 Type II", "controls": 12},
            {"id": "ISO27001", "name": "ISO 27001", "controls": 18}
        ]
    }

# Get controls for a specific framework
@app.get("/frameworks/{framework_id}/controls")
async def get_controls(framework_id: str):
    # Mock data - replace with actual control database
    controls = {
        "SOX": [
            {"id": "SOX-ITGC-01", "name": "Logical Access Controls"},
            {"id": "SOX-ITGC-02", "name": "Change Management"},
            {"id": "SOX-ITGC-03", "name": "Data Backup & Recovery"}
        ],
        "SOC2": [
            {"id": "CC6.1", "name": "Logical and Physical Access Controls"},
            {"id": "CC6.2", "name": "System Credentials"},
            {"id": "CC6.3", "name": "Network Access"}
        ],
        "ISO27001": [
            {"id": "A.9.2.1", "name": "User Registration"},
            {"id": "A.9.2.2", "name": "User Access Provisioning"},
            {"id": "A.9.2.3", "name": "Management of Privileged Access"}
        ]
    }
    
    if framework_id not in controls:
        raise HTTPException(status_code=404, detail="Framework not found")
    
    return {"framework": framework_id, "controls": controls[framework_id]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 