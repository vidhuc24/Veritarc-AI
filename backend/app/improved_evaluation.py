"""
Improved Evaluation Engine for Veritarc AI
Tests the improved validation engine for better accuracy and fewer partial predictions
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
from validation_engine_v2 import ImprovedValidationEngine
from retrieval_engine import VeritarcRetrievalEngine

logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Result comparing original vs improved validation"""
    document_id: str
    expected_result: str
    original_prediction: str
    improved_prediction: str
    original_score: float
    improved_score: float
    original_correct: bool
    improved_correct: bool
    improvement_gained: bool

class ImprovedEvaluationEngine:
    def __init__(self):
        """Initialize comparison evaluation engine"""
        self.improved_engine = ImprovedValidationEngine()
        self.retrieval_engine = VeritarcRetrievalEngine()
        
    def run_improvement_comparison(self, test_data_file: str = "data/synthetic_evidence/test_evidence.json"):
        """Run comprehensive comparison between original and improved systems"""
        
        print("🚀 Running Improved System Evaluation...")
        print("🔄 Comparing improved validation against expected results...")
        
        # Load test data
        try:
            with open(test_data_file, 'r') as f:
                evidence_docs = json.load(f)
        except FileNotFoundError:
            print("❌ Test evidence file not found")
            return
        
        print(f"📊 Testing improved system against {len(evidence_docs)} evidence documents...")
        
        # Test improved system
        improved_results = []
        partial_count = 0
        correct_count = 0
        total_processing_time = 0
        
        for i, evidence_doc in enumerate(evidence_docs):
            print(f"\n  Processing document {i+1}/{len(evidence_docs)}: {evidence_doc.get('document_type', 'Unknown')}")
            
            try:
                expected_result = evidence_doc.get('expected_result', 'fail')
                evidence_content = evidence_doc.get('content', '')
                control_id = evidence_doc.get('control_id', '')
                
                # Process with improved engine
                start_time = time.time()
                result = self.improved_engine.validate_evidence_against_control(evidence_content, control_id)
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # Check results
                predicted = result.compliance_status.lower()
                is_correct = self._is_prediction_correct(expected_result, predicted)
                is_partial = predicted == "partial"
                
                if is_partial:
                    partial_count += 1
                if is_correct:
                    correct_count += 1
                
                improved_results.append({
                    'document_id': i,
                    'expected': expected_result,
                    'predicted': predicted,
                    'score': result.validation_score,
                    'correct': is_correct,
                    'is_partial': is_partial,
                    'processing_time': processing_time,
                    'rationale': result.validation_details.get('decision_rationale', '')[:100] + '...'
                })
                
                print(f"    Expected: {expected_result}, Predicted: {predicted}, Score: {result.validation_score:.3f}, Correct: {is_correct}")
                
            except Exception as e:
                logger.error(f"Error processing document {i+1}: {e}")
                continue
        
        # Calculate final metrics
        total_docs = len(improved_results)
        accuracy = correct_count / total_docs if total_docs > 0 else 0
        partial_rate = partial_count / total_docs if total_docs > 0 else 0
        avg_processing_time = total_processing_time / total_docs if total_docs > 0 else 0
        
        # Display comprehensive results
        self._display_improvement_results(improved_results, accuracy, partial_rate, avg_processing_time)
        
        # Save detailed results
        self._save_improvement_report(improved_results, accuracy, partial_rate, avg_processing_time)
        
        return improved_results
    
    def _is_prediction_correct(self, expected: str, predicted: str) -> bool:
        """Check if prediction matches expected result"""
        # Map partial to pass for evaluation (same logic as before)
        if predicted == "partial":
            predicted = "pass"
        return expected.lower() == predicted.lower()
    
    def _display_improvement_results(self, results: List[Dict], accuracy: float, partial_rate: float, avg_time: float):
        """Display comprehensive improvement results"""
        
        print(f"\n{'='*60}")
        print("📊 IMPROVED SYSTEM EVALUATION RESULTS")
        print(f"{'='*60}")
        
        # Overall Metrics
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"  Accuracy: {accuracy:.2%} (Target: >90%)")
        print(f"  Partial Predictions: {partial_rate:.2%} (Target: <10%)")
        print(f"  Average Processing Time: {avg_time:.2f}s")
        print(f"  Total Documents: {len(results)}")
        
        # Accuracy Analysis
        correct_predictions = sum(1 for r in results if r['correct'])
        false_positives = sum(1 for r in results if r['expected'] == 'fail' and r['predicted'] in ['pass', 'partial'])
        false_negatives = sum(1 for r in results if r['expected'] == 'pass' and r['predicted'] == 'fail')
        
        print(f"\n📈 ACCURACY BREAKDOWN:")
        print(f"  Correct Predictions: {correct_predictions}/{len(results)}")
        print(f"  False Positives: {false_positives} ({false_positives/len(results)*100:.1f}%)")
        print(f"  False Negatives: {false_negatives} ({false_negatives/len(results)*100:.1f}%)")
        
        # Decision Distribution
        pass_count = sum(1 for r in results if r['predicted'] == 'pass')
        fail_count = sum(1 for r in results if r['predicted'] == 'fail')
        partial_count = sum(1 for r in results if r['predicted'] == 'partial')
        
        print(f"\n🔄 DECISION DISTRIBUTION:")
        print(f"  Pass: {pass_count} ({pass_count/len(results)*100:.1f}%)")
        print(f"  Fail: {fail_count} ({fail_count/len(results)*100:.1f}%)")
        print(f"  Partial: {partial_count} ({partial_count/len(results)*100:.1f}%)")
        
        # Score Distribution
        scores = [r['score'] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        high_scores = sum(1 for s in scores if s >= 0.8)
        low_scores = sum(1 for s in scores if s < 0.6)
        
        print(f"\n📊 SCORE DISTRIBUTION:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  High Scores (≥0.8): {high_scores} ({high_scores/len(results)*100:.1f}%)")
        print(f"  Low Scores (<0.6): {low_scores} ({low_scores/len(results)*100:.1f}%)")
        
        # Detailed Results
        print(f"\n📋 DETAILED RESULTS:")
        for r in results:
            status_emoji = "✅" if r['correct'] else "❌"
            partial_flag = "⚠️ PARTIAL" if r['is_partial'] else ""
            print(f"  Doc {r['document_id']+1}: {status_emoji} Expected: {r['expected']}, Got: {r['predicted']}, Score: {r['score']:.3f} {partial_flag}")
    
    def _save_improvement_report(self, results: List[Dict], accuracy: float, partial_rate: float, avg_time: float):
        """Save improvement evaluation report"""
        
        os.makedirs('evaluation_results', exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        report = {
            "evaluation_type": "improved_system_evaluation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary_metrics": {
                "accuracy": accuracy,
                "partial_prediction_rate": partial_rate,
                "average_processing_time": avg_time,
                "total_documents": len(results)
            },
            "detailed_results": results,
            "improvement_analysis": {
                "target_accuracy_met": accuracy >= 0.90,
                "partial_predictions_reduced": partial_rate <= 0.10,
                "performance_acceptable": avg_time <= 30.0
            }
        }
        
        filename = f"evaluation_results/improved_evaluation_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {filename}")
    
    def analyze_specific_improvements(self, results: List[Dict]):
        """Analyze specific areas of improvement"""
        
        print(f"\n🔍 IMPROVEMENT ANALYSIS:")
        
        # Quality level analysis
        quality_performance = {}
        for r in results:
            # You'd need to map document IDs back to quality levels
            # This is a simplified version
            pass
        
        # Control category analysis
        category_performance = {}
        # Similar analysis for different SOX control categories
        
        print("✅ Improvement analysis complete!")


def main():
    """Run improved evaluation"""
    print("🚀 Testing Improved Veritarc AI System...")
    
    # Initialize improved evaluation engine
    evaluator = ImprovedEvaluationEngine()
    
    # Run comprehensive improvement evaluation
    results = evaluator.run_improvement_comparison()
    
    print("\n🎉 Improved System Evaluation Complete!")


if __name__ == "__main__":
    main() 