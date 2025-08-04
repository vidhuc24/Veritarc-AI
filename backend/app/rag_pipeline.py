"""
Main RAG Pipeline for Veritarc AI
End-to-end evidence validation system combining retrieval and validation
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import os

# Add the backend/app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))
from retrieval_engine import VeritarcRetrievalEngine, RetrievalResult
from validation_engine import VeritarcValidationEngine, ValidationResult

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/veritarc-ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Complete result from the RAG pipeline"""
    request_id: str
    evidence_content: str
    control_id: Optional[str]
    retrieval_results: List[RetrievalResult]
    validation_results: List[ValidationResult]
    pipeline_metrics: Dict[str, Any]
    processing_time: float
    status: str  # "success", "partial", "error"

class VeritarcRAGPipeline:
    def __init__(self):
        """Initialize the complete RAG pipeline"""
        logger.info("Initializing Veritarc RAG Pipeline...")
        
        try:
            self.retrieval_engine = VeritarcRetrievalEngine()
            self.validation_engine = VeritarcValidationEngine()
            logger.info("✅ RAG Pipeline components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing RAG Pipeline: {e}")
            raise
    
    def process_evidence(self, evidence_content: str, control_id: Optional[str] = None, 
                        auto_match_controls: bool = True) -> PipelineResult:
        """
        Main pipeline method: Process evidence through retrieval and validation
        
        Args:
            evidence_content: The evidence document content to validate
            control_id: Specific SOX control ID to validate against (optional)
            auto_match_controls: Whether to auto-match controls if control_id not provided
        
        Returns:
            PipelineResult with complete validation results
        """
        
        start_time = time.time()
        request_id = f"req_{int(time.time())}_{hash(evidence_content) % 10000}"
        
        logger.info(f"Processing evidence request {request_id}")
        
        try:
            # Step 1: Retrieve relevant controls
            retrieval_results = self._retrieve_relevant_controls(evidence_content, control_id, auto_match_controls)
            logger.info(f"Retrieved {len(retrieval_results)} relevant controls")
            
            # Step 2: Validate evidence against controls
            validation_results = self._validate_evidence_against_controls(evidence_content, retrieval_results)
            logger.info(f"Completed {len(validation_results)} validations")
            
            # Step 3: Calculate pipeline metrics
            pipeline_metrics = self._calculate_pipeline_metrics(retrieval_results, validation_results)
            
            # Step 4: Determine overall status
            status = self._determine_pipeline_status(validation_results)
            
            processing_time = time.time() - start_time
            
            result = PipelineResult(
                request_id=request_id,
                evidence_content=evidence_content[:500] + "..." if len(evidence_content) > 500 else evidence_content,
                control_id=control_id,
                retrieval_results=retrieval_results,
                validation_results=validation_results,
                pipeline_metrics=pipeline_metrics,
                processing_time=processing_time,
                status=status
            )
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}")
            processing_time = time.time() - start_time
            
            # Return error result
            return PipelineResult(
                request_id=request_id,
                evidence_content=evidence_content[:500] + "..." if len(evidence_content) > 500 else evidence_content,
                control_id=control_id,
                retrieval_results=[],
                validation_results=[],
                pipeline_metrics={"error": str(e)},
                processing_time=processing_time,
                status="error"
            )
    
    def _retrieve_relevant_controls(self, evidence_content: str, control_id: Optional[str], 
                                  auto_match_controls: bool) -> List[RetrievalResult]:
        """Retrieve relevant controls for evidence validation"""
        
        if control_id:
            # Use specific control if provided
            control_details = self.retrieval_engine.get_control_details(control_id)
            if control_details:
                # Create a RetrievalResult for the specific control
                control_docs = self.retrieval_engine.controls_collection.get(ids=[control_id])
                if control_docs['documents']:
                    result = RetrievalResult(
                        document_id=control_id,
                        content=control_docs['documents'][0],
                        metadata=control_details,
                        similarity_score=0.0,  # Perfect match for specific control
                        retrieval_method="specific_control"
                    )
                    return [result]
        
        if auto_match_controls:
            # Auto-match controls using retrieval engine
            return self.retrieval_engine.retrieve_controls_for_evidence(evidence_content)
        
        return []
    
    def _validate_evidence_against_controls(self, evidence_content: str, 
                                          retrieval_results: List[RetrievalResult]) -> List[ValidationResult]:
        """Validate evidence against retrieved controls"""
        
        validation_results = []
        
        for retrieval_result in retrieval_results:
            try:
                control_id = retrieval_result.metadata.get('control_id')
                if control_id:
                    validation_result = self.validation_engine.validate_evidence_against_control(
                        evidence_content, control_id
                    )
                    validation_results.append(validation_result)
                    logger.debug(f"Validated against control {control_id}: score {validation_result.validation_score:.3f}")
            except Exception as e:
                logger.warning(f"Error validating against control {retrieval_result.document_id}: {e}")
                continue
        
        return validation_results
    
    def _calculate_pipeline_metrics(self, retrieval_results: List[RetrievalResult], 
                                  validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Calculate comprehensive pipeline metrics"""
        
        metrics = {
            "retrieval_metrics": {
                "total_controls_retrieved": len(retrieval_results),
                "retrieval_methods": {},
                "average_similarity_score": 0.0
            },
            "validation_metrics": self.validation_engine.calculate_validation_metrics(validation_results),
            "pipeline_metrics": {
                "total_processing_time": 0.0,
                "success_rate": 0.0
            }
        }
        
        # Calculate retrieval metrics
        if retrieval_results:
            similarity_scores = [1 - r.similarity_score for r in retrieval_results]
            metrics["retrieval_metrics"]["average_similarity_score"] = sum(similarity_scores) / len(similarity_scores)
            
            # Count retrieval methods
            for result in retrieval_results:
                method = result.retrieval_method
                metrics["retrieval_metrics"]["retrieval_methods"][method] = \
                    metrics["retrieval_metrics"]["retrieval_methods"].get(method, 0) + 1
        
        return metrics
    
    def _determine_pipeline_status(self, validation_results: List[ValidationResult]) -> str:
        """Determine overall pipeline status based on validation results"""
        
        if not validation_results:
            return "error"
        
        # Check validation results
        pass_count = sum(1 for r in validation_results if r.compliance_status == "Pass")
        fail_count = sum(1 for r in validation_results if r.compliance_status == "Fail")
        partial_count = sum(1 for r in validation_results if r.compliance_status == "Partial")
        
        # Consider both Pass and Partial as successful outcomes
        if pass_count > 0 or partial_count > 0:
            if fail_count == 0:
                return "success"
            else:
                return "partial"
        else:
            return "error"
    
    def batch_process_evidence(self, evidence_documents: List[Dict[str, Any]]) -> List[PipelineResult]:
        """Process multiple evidence documents in batch"""
        
        logger.info(f"Starting batch processing of {len(evidence_documents)} evidence documents")
        
        results = []
        for i, evidence_doc in enumerate(evidence_documents):
            logger.info(f"Processing evidence {i+1}/{len(evidence_documents)}")
            
            try:
                evidence_content = evidence_doc.get('content', '')
                control_id = evidence_doc.get('control_id')
                
                result = self.process_evidence(evidence_content, control_id)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing evidence {i+1}: {e}")
                continue
        
        logger.info(f"Batch processing completed: {len(results)} successful results")
        return results
    
    def get_pipeline_summary(self, results: List[PipelineResult]) -> Dict[str, Any]:
        """Generate summary statistics for pipeline results"""
        
        if not results:
            return {"error": "No results to summarize"}
        
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r.status == "success")
        partial_requests = sum(1 for r in results if r.status == "partial")
        error_requests = sum(1 for r in results if r.status == "error")
        
        avg_processing_time = sum(r.processing_time for r in results) / total_requests
        
        # Aggregate validation metrics
        all_validation_results = []
        for result in results:
            all_validation_results.extend(result.validation_results)
        
        validation_summary = self.validation_engine.calculate_validation_metrics(all_validation_results)
        
        return {
            "total_requests": total_requests,
            "success_rate": successful_requests / total_requests,
            "partial_rate": partial_requests / total_requests,
            "error_rate": error_requests / total_requests,
            "average_processing_time": avg_processing_time,
            "validation_summary": validation_summary
        }
    
    def test_pipeline(self):
        """Test the complete RAG pipeline"""
        print("\n🧪 Testing Complete RAG Pipeline...")
        
        # Load sample evidence
        try:
            with open('data/synthetic_evidence/test_evidence.json', 'r') as f:
                evidence_docs = json.load(f)
        except FileNotFoundError:
            print("❌ Test evidence file not found")
            return
        
        # Test 1: Single evidence processing
        print("Test 1: Processing single evidence document...")
        sample_evidence = evidence_docs[0]
        evidence_content = sample_evidence['content']
        control_id = sample_evidence.get('control_id')
        
        result = self.process_evidence(evidence_content, control_id)
        print(f"Pipeline Result:")
        print(f"  Request ID: {result.request_id}")
        print(f"  Status: {result.status}")
        print(f"  Processing Time: {result.processing_time:.2f}s")
        print(f"  Controls Retrieved: {len(result.retrieval_results)}")
        print(f"  Validations Completed: {len(result.validation_results)}")
        
        if result.validation_results:
            best_validation = max(result.validation_results, key=lambda x: x.validation_score)
            print(f"  Best Validation Score: {best_validation.validation_score:.3f}")
            print(f"  Compliance Status: {best_validation.compliance_status}")
        
        # Test 2: Batch processing
        print("\nTest 2: Batch processing evidence documents...")
        batch_results = self.batch_process_evidence(evidence_docs[:3])
        
        print(f"Batch processing completed: {len(batch_results)} results")
        
        # Test 3: Pipeline summary
        print("\nTest 3: Pipeline summary statistics...")
        summary = self.get_pipeline_summary(batch_results)
        print(f"Pipeline Summary:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Average Processing Time: {summary['average_processing_time']:.2f}s")
        print(f"  Validation Pass Rate: {summary['validation_summary'].get('pass_rate', 0):.2%}")
        
        print("✅ RAG Pipeline tests completed!")


def main():
    """Test the complete RAG pipeline"""
    print("🚀 Testing Veritarc AI Complete RAG Pipeline...")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Initialize pipeline
    pipeline = VeritarcRAGPipeline()
    
    # Test pipeline functionality
    pipeline.test_pipeline()
    
    print("\n🎉 Step 4 Complete: RAG Pipeline Integration Successful!")


if __name__ == "__main__":
    main() 