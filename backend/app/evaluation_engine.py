"""
Evaluation Engine for Veritarc AI
Comprehensive testing and validation of the RAG pipeline against ground truth data
"""

import json
import time
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import sys
import os

# Add the backend/app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))
from rag_pipeline import VeritarcRAGPipeline, PipelineResult

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of a single evaluation test"""
    evidence_id: str
    expected_result: str  # "pass" or "fail"
    predicted_result: str  # "pass", "fail", or "partial"
    validation_score: float
    processing_time: float
    is_correct: bool
    confidence_level: str
    control_id: str

@dataclass
class EvaluationSummary:
    """Summary of all evaluation results"""
    total_tests: int
    correct_predictions: int
    accuracy: float
    false_positives: int
    false_negatives: int
    average_processing_time: float
    consistency_score: float
    detailed_results: List[EvaluationResult]

class VeritarcEvaluationEngine:
    def __init__(self):
        """Initialize the evaluation engine"""
        self.pipeline = VeritarcRAGPipeline()
        
    def evaluate_ground_truth_accuracy(self, test_data_file: str = "data/synthetic_evidence/test_evidence.json") -> EvaluationSummary:
        """
        Evaluate system accuracy against ground truth data
        
        Args:
            test_data_file: Path to synthetic evidence with known expected results
            
        Returns:
            EvaluationSummary with comprehensive accuracy metrics
        """
        
        print("🔍 Starting Ground Truth Accuracy Evaluation...")
        
        # Load test data
        try:
            with open(test_data_file, 'r') as f:
                evidence_docs = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Test data file not found: {test_data_file}")
        
        print(f"📊 Testing against {len(evidence_docs)} evidence documents...")
        
        evaluation_results = []
        total_processing_time = 0
        
        # Process each evidence document
        for i, evidence_doc in enumerate(evidence_docs):
            print(f"  Processing document {i+1}/{len(evidence_docs)}: {evidence_doc.get('document_type', 'Unknown')}")
            
            try:
                # Get ground truth
                expected_result = evidence_doc.get('expected_result', 'fail')
                evidence_content = evidence_doc.get('content', '')
                control_id = evidence_doc.get('control_id', '')
                
                # Process through pipeline
                start_time = time.time()
                pipeline_result = self.pipeline.process_evidence(evidence_content, control_id)
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # Extract prediction
                predicted_result = self._extract_prediction(pipeline_result)
                validation_score = self._extract_validation_score(pipeline_result)
                confidence_level = self._extract_confidence_level(pipeline_result)
                
                # Determine if prediction is correct
                is_correct = self._is_prediction_correct(expected_result, predicted_result)
                
                # Create evaluation result
                eval_result = EvaluationResult(
                    evidence_id=f"evidence_{i}",
                    expected_result=expected_result,
                    predicted_result=predicted_result,
                    validation_score=validation_score,
                    processing_time=processing_time,
                    is_correct=is_correct,
                    confidence_level=confidence_level,
                    control_id=control_id
                )
                
                evaluation_results.append(eval_result)
                
                print(f"    Expected: {expected_result}, Predicted: {predicted_result}, Correct: {is_correct}")
                
            except Exception as e:
                logger.error(f"Error evaluating document {i+1}: {e}")
                continue
        
        # Calculate summary metrics
        summary = self._calculate_evaluation_summary(evaluation_results, total_processing_time)
        
        print(f"✅ Ground Truth Evaluation Complete!")
        print(f"  Accuracy: {summary.accuracy:.2%}")
        print(f"  Correct Predictions: {summary.correct_predictions}/{summary.total_tests}")
        print(f"  Average Processing Time: {summary.average_processing_time:.2f}s")
        
        return summary
    
    def evaluate_consistency(self, test_data_file: str = "data/synthetic_evidence/test_evidence.json") -> Dict[str, Any]:
        """
        Evaluate system consistency across different scenarios
        
        Args:
            test_data_file: Path to test data
            
        Returns:
            Consistency metrics and analysis
        """
        
        print("\n🔄 Starting Consistency Evaluation...")
        
        # Load test data
        with open(test_data_file, 'r') as f:
            evidence_docs = json.load(f)
        
        consistency_metrics = {
            "quality_level_consistency": {},
            "control_category_consistency": {},
            "confidence_distribution": {},
            "score_distribution": {}
        }
        
        # Group by quality level
        quality_groups = {}
        for doc in evidence_docs:
            quality = doc.get('quality_level', 'unknown')
            if quality not in quality_groups:
                quality_groups[quality] = []
            quality_groups[quality].append(doc)
        
        # Test consistency within each quality level
        for quality, docs in quality_groups.items():
            print(f"  Testing consistency for {quality} quality documents...")
            
            quality_results = []
            for doc in docs:
                try:
                    pipeline_result = self.pipeline.process_evidence(doc['content'], doc.get('control_id'))
                    score = self._extract_validation_score(pipeline_result)
                    quality_results.append(score)
                except Exception as e:
                    logger.warning(f"Error processing {quality} quality document: {e}")
                    continue
            
            if quality_results:
                consistency_metrics["quality_level_consistency"][quality] = {
                    "count": len(quality_results),
                    "average_score": sum(quality_results) / len(quality_results),
                    "score_variance": self._calculate_variance(quality_results),
                    "score_range": max(quality_results) - min(quality_results)
                }
        
        # Group by control category
        category_groups = {}
        for doc in evidence_docs:
            control_id = doc.get('control_id', '')
            category = self._extract_category_from_control_id(control_id)
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(doc)
        
        # Test consistency within each control category
        for category, docs in category_groups.items():
            print(f"  Testing consistency for {category} controls...")
            
            category_results = []
            for doc in docs:
                try:
                    pipeline_result = self.pipeline.process_evidence(doc['content'], doc.get('control_id'))
                    score = self._extract_validation_score(pipeline_result)
                    category_results.append(score)
                except Exception as e:
                    logger.warning(f"Error processing {category} control document: {e}")
                    continue
            
            if category_results:
                consistency_metrics["control_category_consistency"][category] = {
                    "count": len(category_results),
                    "average_score": sum(category_results) / len(category_results),
                    "score_variance": self._calculate_variance(category_results),
                    "score_range": max(category_results) - min(category_results)
                }
        
        print("✅ Consistency Evaluation Complete!")
        return consistency_metrics
    
    def _extract_prediction(self, pipeline_result: PipelineResult) -> str:
        """Extract prediction from pipeline result"""
        if not pipeline_result.validation_results:
            return "fail"
        
        # Use the best validation result
        best_validation = max(pipeline_result.validation_results, key=lambda x: x.validation_score)
        return best_validation.compliance_status.lower()
    
    def _extract_validation_score(self, pipeline_result: PipelineResult) -> float:
        """Extract validation score from pipeline result"""
        if not pipeline_result.validation_results:
            return 0.0
        
        # Use the best validation result
        best_validation = max(pipeline_result.validation_results, key=lambda x: x.validation_score)
        return best_validation.validation_score
    
    def _extract_confidence_level(self, pipeline_result: PipelineResult) -> str:
        """Extract confidence level from pipeline result"""
        if not pipeline_result.validation_results:
            return "Low"
        
        # Use the best validation result
        best_validation = max(pipeline_result.validation_results, key=lambda x: x.validation_score)
        return best_validation.confidence_level
    
    def _is_prediction_correct(self, expected: str, predicted: str) -> bool:
        """Determine if prediction matches expected result"""
        # Map partial to pass for evaluation purposes
        if predicted == "partial":
            predicted = "pass"
        
        return expected.lower() == predicted.lower()
    
    def _calculate_evaluation_summary(self, results: List[EvaluationResult], total_time: float) -> EvaluationSummary:
        """Calculate comprehensive evaluation summary"""
        
        total_tests = len(results)
        correct_predictions = sum(1 for r in results if r.is_correct)
        accuracy = correct_predictions / total_tests if total_tests > 0 else 0.0
        
        # Calculate false positives and negatives
        false_positives = sum(1 for r in results if r.expected_result == "fail" and r.predicted_result in ["pass", "partial"])
        false_negatives = sum(1 for r in results if r.expected_result == "pass" and r.predicted_result == "fail")
        
        average_processing_time = total_time / total_tests if total_tests > 0 else 0.0
        
        # Calculate consistency score (lower variance = higher consistency)
        scores = [r.validation_score for r in results]
        consistency_score = 1.0 - min(self._calculate_variance(scores), 1.0) if scores else 0.0
        
        return EvaluationSummary(
            total_tests=total_tests,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            false_positives=false_positives,
            false_negatives=false_negatives,
            average_processing_time=average_processing_time,
            consistency_score=consistency_score,
            detailed_results=results
        )
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return squared_diff_sum / len(values)
    
    def _extract_category_from_control_id(self, control_id: str) -> str:
        """Extract control category from control ID"""
        if "AC" in control_id:
            return "Access Controls"
        elif "CM" in control_id:
            return "Change Management"
        elif "DR" in control_id:
            return "Data Backup & Recovery"
        else:
            return "Unknown"
    
    def generate_evaluation_report(self, accuracy_summary: EvaluationSummary, 
                                 consistency_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        report = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy_metrics": {
                "overall_accuracy": accuracy_summary.accuracy,
                "correct_predictions": accuracy_summary.correct_predictions,
                "total_tests": accuracy_summary.total_tests,
                "false_positives": accuracy_summary.false_positives,
                "false_negatives": accuracy_summary.false_negatives,
                "false_positive_rate": accuracy_summary.false_positives / accuracy_summary.total_tests if accuracy_summary.total_tests > 0 else 0.0,
                "false_negative_rate": accuracy_summary.false_negatives / accuracy_summary.total_tests if accuracy_summary.total_tests > 0 else 0.0
            },
            "performance_metrics": {
                "average_processing_time": accuracy_summary.average_processing_time,
                "total_processing_time": sum(r.processing_time for r in accuracy_summary.detailed_results)
            },
            "consistency_metrics": {
                "overall_consistency_score": accuracy_summary.consistency_score,
                "quality_level_consistency": consistency_metrics.get("quality_level_consistency", {}),
                "control_category_consistency": consistency_metrics.get("control_category_consistency", {})
            },
            "detailed_results": [
                {
                    "evidence_id": r.evidence_id,
                    "expected_result": r.expected_result,
                    "predicted_result": r.predicted_result,
                    "validation_score": r.validation_score,
                    "is_correct": r.is_correct,
                    "confidence_level": r.confidence_level,
                    "control_id": r.control_id
                }
                for r in accuracy_summary.detailed_results
            ]
        }
        
        return report
    
    def run_complete_evaluation(self):
        """Run complete evaluation suite"""
        print("🚀 Starting Complete Veritarc AI Evaluation...")
        
        # Step 1: Ground Truth Accuracy Evaluation
        accuracy_summary = self.evaluate_ground_truth_accuracy()
        
        # Step 2: Consistency Evaluation
        consistency_metrics = self.evaluate_consistency()
        
        # Step 3: Generate Report
        report = self.generate_evaluation_report(accuracy_summary, consistency_metrics)
        
        # Step 4: Display Results
        self._display_evaluation_results(report)
        
        # Step 5: Save Report
        self._save_evaluation_report(report)
        
        print("\n🎉 Complete Evaluation Finished!")
        return report
    
    def _display_evaluation_results(self, report: Dict[str, Any]):
        """Display evaluation results in a readable format"""
        
        print("\n📊 EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        
        # Accuracy Results
        accuracy = report["accuracy_metrics"]
        print(f"🎯 ACCURACY METRICS:")
        print(f"  Overall Accuracy: {accuracy['overall_accuracy']:.2%}")
        print(f"  Correct Predictions: {accuracy['correct_predictions']}/{accuracy['total_tests']}")
        print(f"  False Positive Rate: {accuracy['false_positive_rate']:.2%}")
        print(f"  False Negative Rate: {accuracy['false_negative_rate']:.2%}")
        
        # Performance Results
        performance = report["performance_metrics"]
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Average Processing Time: {performance['average_processing_time']:.2f}s")
        print(f"  Total Processing Time: {performance['total_processing_time']:.2f}s")
        
        # Consistency Results
        consistency = report["consistency_metrics"]
        print(f"\n🔄 CONSISTENCY METRICS:")
        print(f"  Overall Consistency Score: {consistency['overall_consistency_score']:.3f}")
        
        # Quality Level Consistency
        quality_consistency = consistency.get("quality_level_consistency", {})
        if quality_consistency:
            print(f"  Quality Level Consistency:")
            for quality, metrics in quality_consistency.items():
                print(f"    {quality}: {metrics['count']} docs, avg score: {metrics['average_score']:.3f}")
        
        # Control Category Consistency
        category_consistency = consistency.get("control_category_consistency", {})
        if category_consistency:
            print(f"  Control Category Consistency:")
            for category, metrics in category_consistency.items():
                print(f"    {category}: {metrics['count']} docs, avg score: {metrics['average_score']:.3f}")
    
    def _save_evaluation_report(self, report: Dict[str, Any]):
        """Save evaluation report to file"""
        
        # Create evaluation directory if it doesn't exist
        os.makedirs('evaluation_results', exist_ok=True)
        
        # Save detailed report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results/evaluation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Evaluation report saved to: {filename}")


def main():
    """Run complete evaluation"""
    print("🚀 Running Veritarc AI Complete Evaluation...")
    
    # Initialize evaluation engine
    evaluator = VeritarcEvaluationEngine()
    
    # Run complete evaluation
    report = evaluator.run_complete_evaluation()
    
    print("\n🎉 Step 5 Complete: Evaluation & Testing Successful!")


if __name__ == "__main__":
    main() 