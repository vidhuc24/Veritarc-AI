"""
Phase 2 - Step 2a-2: Create RAGAS Evaluation Dataset
Combines original synthetic evidence with enhanced RAGAS evidence for comprehensive evaluation
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime
import sys

# Add backend path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))
from vector_store import VeritarcVectorStore
from retrieval_engine import VeritarcRetrievalEngine

def load_original_evidence() -> List[Dict[str, Any]]:
    """Load the original 9 synthetic evidence documents"""
    print("📋 Loading original synthetic evidence...")
    
    with open('data/synthetic_evidence/test_evidence.json', 'r') as f:
        original_evidence = json.load(f)
    
    print(f"✅ Loaded {len(original_evidence)} original evidence documents")
    return original_evidence

def load_enhanced_evidence() -> List[Dict[str, Any]]:
    """Load the 3 enhanced RAGAS evidence documents"""
    print("📋 Loading enhanced RAGAS evidence...")
    
    with open('data/enhanced_evidence/ragas_enhanced_evidence.json', 'r') as f:
        enhanced_evidence = json.load(f)
    
    print(f"✅ Loaded {len(enhanced_evidence)} enhanced evidence documents")
    return enhanced_evidence

def load_sox_controls() -> List[Dict[str, Any]]:
    """Load SOX controls for reference"""
    print("📋 Loading SOX controls...")
    
    with open('data/sample_sox_controls.json', 'r') as f:
        controls = json.load(f)
    
    print(f"✅ Loaded {len(controls)} SOX controls")
    return controls

def create_ragas_evaluation_questions(evidence_docs: List[Dict[str, Any]], 
                                    controls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create RAGAS-style evaluation questions from evidence documents"""
    print("🔍 Creating RAGAS evaluation questions...")
    
    evaluation_questions = []
    
    for i, evidence in enumerate(evidence_docs):
        # Find the corresponding control
        control_id = evidence['control_id']
        control = next((c for c in controls if c['control_id'] == control_id), None)
        
        if not control:
            print(f"❌ Warning: Control {control_id} not found for evidence {i}")
            continue
        
        # Create evaluation question based on evidence type
        question_templates = {
            'access_review_report': f"Does this access review report demonstrate compliance with {control['name']}? Evaluate the evidence for proper user access management, approval processes, and documentation.",
            'change_request_form': f"Does this change request form meet the requirements of {control['name']}? Assess the approval workflow, testing procedures, and documentation completeness.",
            'backup_verification_log': f"Does this backup verification log satisfy {control['name']} requirements? Review the backup procedures, verification steps, and recovery testing evidence."
        }
        
        # Get appropriate question template
        doc_type = evidence['document_type']
        question = question_templates.get(doc_type, 
            f"Does this {doc_type} demonstrate compliance with {control['name']}? Evaluate the evidence against the control requirements.")
        
        # Create ground truth answer based on expected result
        ground_truth = "PASS" if evidence['expected_result'] == 'pass' else "FAIL"
        
        # Create detailed context from control
        context = f"""
Control: {control['control_id']} - {control['name']}
Category: {control['category']}
Description: {control['description']}

Validation Criteria:
{chr(10).join(f"• {criteria}" for criteria in control['validation_criteria'])}

Keywords: {', '.join(control['keywords'])}
"""
        
        evaluation_item = {
            "question": question,
            "ground_truth": ground_truth,
            "contexts": [context],
            "evidence_content": evidence['content'],
            "evidence_metadata": {
                "evidence_id": f"evidence_{i}_{doc_type}",
                "control_id": control_id,
                "document_type": doc_type,
                "company": evidence.get('company', 'Unknown'),
                "quality_level": evidence.get('quality_level', 'unknown'),
                "expected_result": evidence['expected_result'],
                "dataset_source": evidence.get('enhancements_applied', ['original'])[0] if evidence.get('enhancements_applied') else 'original'
            }
        }
        
        evaluation_questions.append(evaluation_item)
    
    print(f"✅ Created {len(evaluation_questions)} evaluation questions")
    return evaluation_questions

def create_rag_pipeline_dataset(evaluation_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create dataset specifically for RAG pipeline evaluation"""
    print("🔄 Creating RAG pipeline evaluation dataset...")
    
    # Initialize RAG components
    retrieval_engine = VeritarcRetrievalEngine()
    
    rag_dataset = []
    
    for item in evaluation_questions:
        evidence_content = item['evidence_content']
        control_id = item['evidence_metadata']['control_id']
        
        # Use retrieval engine to get relevant controls
        retrieved_controls = retrieval_engine.retrieve_controls_for_evidence(evidence_content)
        
        # Create contexts from retrieved controls
        retrieved_contexts = []
        for control_result in retrieved_controls:
            context = f"""
Control: {control_result.metadata.get('control_id')} - {control_result.metadata.get('name')}
Category: {control_result.metadata.get('category')}
Description: {control_result.metadata.get('description', '')}
Similarity Score: {control_result.similarity_score:.3f}
"""
            retrieved_contexts.append(context)
        
        rag_item = {
            "question": item['question'],
            "ground_truth": item['ground_truth'],
            "contexts": retrieved_contexts,  # Use retrieved contexts instead of ground truth
            "ground_truth_contexts": item['contexts'],  # Keep original for comparison
            "evidence_content": evidence_content,
            "evidence_metadata": item['evidence_metadata'],
            "retrieval_results": {
                "num_retrieved": len(retrieved_controls),
                "top_control_id": retrieved_controls[0].document_id if retrieved_controls else None,
                "top_similarity": retrieved_controls[0].similarity_score if retrieved_controls else 0.0,
                "correct_control_retrieved": any(r.document_id == control_id for r in retrieved_controls)
            }
        }
        
        rag_dataset.append(rag_item)
    
    print(f"✅ Created RAG pipeline dataset with {len(rag_dataset)} items")
    return rag_dataset

def save_evaluation_datasets(evaluation_questions: List[Dict[str, Any]], 
                           rag_dataset: List[Dict[str, Any]]) -> None:
    """Save the evaluation datasets"""
    print("💾 Saving evaluation datasets...")
    
    # Create output directory
    os.makedirs('data/evaluation_datasets', exist_ok=True)
    
    # Save basic evaluation questions
    with open('data/evaluation_datasets/ragas_evaluation_questions.json', 'w') as f:
        json.dump(evaluation_questions, f, indent=2)
    
    # Save RAG pipeline dataset
    with open('data/evaluation_datasets/rag_pipeline_dataset.json', 'w') as f:
        json.dump(rag_dataset, f, indent=2)
    
    # Create summary
    summary = {
        "creation_timestamp": datetime.now().isoformat(),
        "datasets_created": {
            "basic_evaluation_questions": {
                "file": "ragas_evaluation_questions.json",
                "count": len(evaluation_questions),
                "description": "Basic RAGAS evaluation questions with ground truth contexts"
            },
            "rag_pipeline_dataset": {
                "file": "rag_pipeline_dataset.json", 
                "count": len(rag_dataset),
                "description": "Full RAG pipeline evaluation with retrieved contexts"
            }
        },
        "dataset_composition": {
            "original_evidence": len([q for q in evaluation_questions if q['evidence_metadata']['dataset_source'] == 'original']),
            "enhanced_evidence": len([q for q in evaluation_questions if q['evidence_metadata']['dataset_source'] != 'original']),
            "total": len(evaluation_questions)
        },
        "quality_distribution": {},
        "control_coverage": {}
    }
    
    # Calculate quality distribution
    for item in evaluation_questions:
        quality = item['evidence_metadata']['quality_level']
        summary['quality_distribution'][quality] = summary['quality_distribution'].get(quality, 0) + 1
    
    # Calculate control coverage
    for item in evaluation_questions:
        control_id = item['evidence_metadata']['control_id']
        summary['control_coverage'][control_id] = summary['control_coverage'].get(control_id, 0) + 1
    
    with open('data/evaluation_datasets/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Saved evaluation datasets:")
    print(f"  📄 Basic questions: data/evaluation_datasets/ragas_evaluation_questions.json")
    print(f"  🔄 RAG pipeline: data/evaluation_datasets/rag_pipeline_dataset.json")
    print(f"  📊 Summary: data/evaluation_datasets/dataset_summary.json")

def test_dataset_creation():
    """Test the dataset creation process"""
    print("🚀 Testing RAGAS Evaluation Dataset Creation")
    print("=" * 60)
    
    try:
        # Load all data
        original_evidence = load_original_evidence()
        enhanced_evidence = load_enhanced_evidence()
        controls = load_sox_controls()
        
        # Combine evidence datasets
        all_evidence = original_evidence + enhanced_evidence
        print(f"\n📊 Combined Dataset: {len(all_evidence)} total evidence documents")
        print(f"  • Original: {len(original_evidence)} documents")
        print(f"  • Enhanced: {len(enhanced_evidence)} documents")
        
        # Create evaluation questions
        evaluation_questions = create_ragas_evaluation_questions(all_evidence, controls)
        
        # Create RAG pipeline dataset
        rag_dataset = create_rag_pipeline_dataset(evaluation_questions)
        
        # Save datasets
        save_evaluation_datasets(evaluation_questions, rag_dataset)
        
        # Display sample
        print(f"\n📋 Sample Evaluation Question:")
        sample = evaluation_questions[0]
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Ground Truth: {sample['ground_truth']}")
        print(f"  Control ID: {sample['evidence_metadata']['control_id']}")
        print(f"  Quality Level: {sample['evidence_metadata']['quality_level']}")
        print(f"  Dataset Source: {sample['evidence_metadata']['dataset_source']}")
        
        print(f"\n📋 Sample RAG Pipeline Item:")
        rag_sample = rag_dataset[0]
        print(f"  Retrieved Controls: {rag_sample['retrieval_results']['num_retrieved']}")
        print(f"  Top Control: {rag_sample['retrieval_results']['top_control_id']}")
        print(f"  Top Similarity: {rag_sample['retrieval_results']['top_similarity']:.3f}")
        print(f"  Correct Retrieved: {rag_sample['retrieval_results']['correct_control_retrieved']}")
        
        print(f"\n✅ Step 2a-2 Complete: RAGAS Evaluation Dataset Created Successfully!")
        
        return evaluation_questions, rag_dataset
        
    except Exception as e:
        print(f"❌ Error in Step 2a-2: {e}")
        raise

if __name__ == "__main__":
    test_dataset_creation() 