"""
Step 1d: Question Analysis & Evidence Mapping Test
Analyze generated RAGAS questions and map them to our evidence document types
"""

import json
import os
import pandas as pd
from typing import List, Dict, Set
from collections import defaultdict
from langchain.schema import Document
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer, 
    MultiHopSpecificQuerySynthesizer
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_sox_testset():
    """Generate SOX testset (from previous steps)"""
    
    print("🔗 Generating SOX testset...")
    
    # Load and prepare documents
    with open("data/sample_sox_controls.json", 'r') as f:
        sox_controls = json.load(f)
    
    documents = []
    for control in sox_controls:
        content = f"""Control ID: {control['control_id']}
Control Name: {control['name']}
Category: {control['category']}

Description:
{control['description']}

Validation Criteria:
{chr(10).join(f"- {criteria}" for criteria in control['validation_criteria'])}

Keywords: {', '.join(control['keywords'])}
"""
        
        doc = Document(
            page_content=content,
            metadata={
                "control_id": control['control_id'],
                "name": control['name'],
                "category": control['category'],
                "document_type": "sox_control"
            }
        )
        documents.append(doc)
    
    # Create knowledge graph
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata
                }
            )
        )
    
    # Apply transformations
    transformer_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
    embedding_model = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    transforms = default_transforms(documents=documents, llm=transformer_llm, embedding_model=embedding_model)
    apply_transforms(kg, transforms)
    
    # Generate testset
    generator = TestsetGenerator(llm=transformer_llm, embedding_model=embedding_model, knowledge_graph=kg)
    
    sox_query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=transformer_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=transformer_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=transformer_llm), 0.25),
    ]
    
    testset = generator.generate(testset_size=8, query_distribution=sox_query_distribution)
    
    print(f"✅ Generated {len(testset)} SOX questions")
    return testset

def analyze_question_themes(testset) -> Dict[str, List[str]]:
    """Analyze question themes and categorize them"""
    
    print("\n🔍 Analyzing Question Themes...")
    
    df = testset.to_pandas()
    
    # Define theme categories based on SOX control areas
    theme_categories = {
        "Access Controls": [
            "access", "user", "authentication", "authorization", "login", "password",
            "permission", "role", "privilege", "account", "identity"
        ],
        "Change Management": [
            "change", "modification", "update", "deployment", "release", "approval",
            "testing", "rollback", "implementation", "development"
        ],
        "Data Backup & Recovery": [
            "backup", "recovery", "restore", "data", "storage", "archive",
            "disaster", "business continuity", "retention"
        ],
        "General Compliance": [
            "audit", "compliance", "validation", "control", "procedure", "policy",
            "documentation", "monitoring", "review", "assessment"
        ]
    }
    
    # Categorize questions by theme
    question_themes = defaultdict(list)
    
    for i, row in df.iterrows():
        question = row['user_input'].lower()
        reference = row.get('reference', '').lower()
        combined_text = f"{question} {reference}"
        
        # Score each theme category
        theme_scores = {}
        for theme, keywords in theme_categories.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                theme_scores[theme] = score
        
        # Assign to best matching theme(s)
        if theme_scores:
            max_score = max(theme_scores.values())
            best_themes = [theme for theme, score in theme_scores.items() if score == max_score]
            for theme in best_themes:
                question_themes[theme].append({
                    'question': row['user_input'],
                    'reference': row.get('reference', ''),
                    'synthesizer': row.get('synthesizer_name', ''),
                    'score': max_score
                })
        else:
            question_themes["General Compliance"].append({
                'question': row['user_input'],
                'reference': row.get('reference', ''),
                'synthesizer': row.get('synthesizer_name', ''),
                'score': 0
            })
    
    # Display analysis
    print(f"📊 Question Theme Distribution:")
    for theme, questions in question_themes.items():
        print(f"  {theme}: {len(questions)} questions")
        if questions:
            sample_q = questions[0]['question'][:80] + "..." if len(questions[0]['question']) > 80 else questions[0]['question']
            print(f"    Sample: {sample_q}")
    
    return dict(question_themes)

def map_questions_to_evidence_types(question_themes: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Map question themes to our evidence document types"""
    
    print(f"\n🗺️  Mapping Questions to Evidence Types...")
    
    # Our evidence document types
    evidence_types = {
        "Access Review Reports": {
            "description": "Periodic reviews of user access rights and permissions",
            "validates": ["User account management", "Access permissions", "Role assignments", "Termination procedures"],
            "maps_to_themes": ["Access Controls"]
        },
        "Change Request Forms": {
            "description": "Documentation for system changes and deployments",
            "validates": ["Change approval process", "Testing documentation", "Rollback procedures", "Implementation timeline"],
            "maps_to_themes": ["Change Management"]
        },
        "Backup Verification Logs": {
            "description": "Evidence of data backup success and integrity verification",
            "validates": ["Backup completion", "Data integrity", "Recovery testing", "Retention compliance"],
            "maps_to_themes": ["Data Backup & Recovery"]
        }
    }
    
    # Create mapping
    question_evidence_mapping = {}
    
    print(f"📋 Question-to-Evidence Mapping:")
    
    for evidence_type, info in evidence_types.items():
        mapped_questions = []
        
        for theme in info["maps_to_themes"]:
            if theme in question_themes:
                mapped_questions.extend(question_themes[theme])
        
        # Also include general compliance questions that could apply
        if evidence_type == "Access Review Reports" and "General Compliance" in question_themes:
            # Add general compliance questions that mention access/audit
            for q in question_themes["General Compliance"]:
                if any(keyword in q['question'].lower() for keyword in ['access', 'user', 'account', 'audit']):
                    mapped_questions.append(q)
        
        question_evidence_mapping[evidence_type] = mapped_questions
        
        print(f"\n  {evidence_type}:")
        print(f"    Description: {info['description']}")
        print(f"    Mapped questions: {len(mapped_questions)}")
        print(f"    Validates: {', '.join(info['validates'])}")
        
        if mapped_questions:
            sample_q = mapped_questions[0]['question'][:100] + "..." if len(mapped_questions[0]['question']) > 100 else mapped_questions[0]['question']
            print(f"    Sample question: {sample_q}")
    
    return question_evidence_mapping

def create_enhancement_recommendations(question_themes: Dict[str, List[str]], 
                                    question_evidence_mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Create recommendations for enhancing our evidence generation"""
    
    print(f"\n💡 Evidence Generation Enhancement Recommendations:")
    
    recommendations = {
        "scenario_enhancement": [],
        "quality_variations": [],
        "realistic_details": []
    }
    
    # Analyze what RAGAS questions reveal about what auditors care about
    for theme, questions in question_themes.items():
        if not questions:
            continue
            
        print(f"\n  {theme} Insights:")
        
        # Extract key concepts from questions
        key_concepts = set()
        for q in questions:
            question_text = q['question'].lower()
            
            # Extract important validation concepts
            if 'validation criteria' in question_text:
                recommendations["quality_variations"].append(f"Include explicit validation criteria testing for {theme}")
            
            if 'specialist' in question_text or 'role' in question_text:
                recommendations["realistic_details"].append(f"Include role-specific details for {theme} evidence")
            
            if 'control id' in question_text:
                recommendations["scenario_enhancement"].append(f"Reference specific control IDs in {theme} evidence")
            
            if any(word in question_text for word in ['procedure', 'process', 'workflow']):
                recommendations["scenario_enhancement"].append(f"Add process workflow details to {theme} evidence")
    
    # Display recommendations
    for category, recs in recommendations.items():
        if recs:
            print(f"\n  {category.replace('_', ' ').title()}:")
            unique_recs = list(set(recs))  # Remove duplicates
            for i, rec in enumerate(unique_recs[:3]):  # Show top 3
                print(f"    {i+1}. {rec}")
    
    return recommendations

def test_question_analysis_and_mapping():
    """Main test function for Step 1d"""
    
    print("🚀 Starting Step 1d: Question Analysis & Evidence Mapping Test")
    print("=" * 70)
    
    try:
        # Step 1: Generate testset
        testset = generate_sox_testset()
        
        # Step 2: Analyze question themes
        question_themes = analyze_question_themes(testset)
        
        # Step 3: Map to evidence types
        question_evidence_mapping = map_questions_to_evidence_types(question_themes)
        
        # Step 4: Create enhancement recommendations
        recommendations = create_enhancement_recommendations(question_themes, question_evidence_mapping)
        
        # Step 5: Verify mapping quality
        total_mapped_questions = sum(len(questions) for questions in question_evidence_mapping.values())
        total_questions = len(testset.to_pandas())
        mapping_coverage = total_mapped_questions / total_questions if total_questions > 0 else 0
        
        print(f"\n✅ Step 1d Test Results:")
        print(f"  ✓ Generated and analyzed {total_questions} questions")
        print(f"  ✓ Identified {len(question_themes)} question theme categories")
        print(f"  ✓ Created mappings for {len(question_evidence_mapping)} evidence types")
        print(f"  ✓ Mapping coverage: {mapping_coverage:.1%} of questions mapped to evidence")
        print(f"  ✓ Generated {sum(len(recs) for recs in recommendations.values())} enhancement recommendations")
        
        print(f"\n🎯 Step 1d: COMPLETE - Ready for Step 1e (Evidence Generation Enhancement)")
        
        return question_themes, question_evidence_mapping, recommendations
        
    except Exception as e:
        print(f"❌ Error in Step 1d: {e}")
        raise


if __name__ == "__main__":
    test_question_analysis_and_mapping() 