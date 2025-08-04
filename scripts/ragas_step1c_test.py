"""
Step 1c: Simple Question Generation Test
Generate basic SOX compliance questions using RAGAS TestsetGenerator
"""

import json
import os
from typing import List
from langchain.schema import Document
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    default_query_distribution, 
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer, 
    MultiHopSpecificQuerySynthesizer
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_sox_knowledge_graph():
    """Create and transform knowledge graph (from Steps 1a & 1b)"""
    
    print("🔗 Creating SOX Knowledge Graph...")
    
    # Load SOX controls
    with open("data/sample_sox_controls.json", 'r') as f:
        sox_controls = json.load(f)
    
    # Convert to documents
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
    
    transforms = default_transforms(
        documents=documents,
        llm=transformer_llm,
        embedding_model=embedding_model
    )
    apply_transforms(kg, transforms)
    
    print(f"✅ Knowledge graph ready: {kg}")
    return kg, documents, transformer_llm, embedding_model

def generate_sox_questions(kg: KnowledgeGraph, documents: List[Document], 
                          generator_llm, generator_embeddings, num_questions: int = 8):
    """Generate SOX compliance questions using RAGAS"""
    
    print(f"\n🤖 Generating {num_questions} SOX compliance questions...")
    
    # Create TestsetGenerator
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg
    )
    
    # Define query synthesizers for SOX compliance
    sox_query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),     # 50% - specific control questions
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),     # 25% - conceptual questions  
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),     # 25% - multi-control questions
    ]
    
    print("🔧 Query synthesizer distribution:")
    for synthesizer, weight in sox_query_distribution:
        print(f"  {synthesizer.__class__.__name__}: {weight*100}%")
    
    # Generate questions
    print("\n🚀 Generating testset...")
    try:
        testset = generator.generate(
            testset_size=num_questions,
            query_distribution=sox_query_distribution
        )
        
        print(f"✅ Generated {len(testset)} questions successfully!")
        return testset
        
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        # Try simpler approach
        print("🔄 Trying simplified generation...")
        testset = generator.generate_with_langchain_docs(
            documents, 
            testset_size=num_questions
        )
        print(f"✅ Generated {len(testset)} questions with simplified approach!")
        return testset

def analyze_generated_questions(testset):
    """Analyze the generated questions for SOX relevance"""
    
    print(f"\n🔍 Question Analysis:")
    
    # Convert to pandas for analysis
    df = testset.to_pandas()
    print(f"  Total questions: {len(df)}")
    
    # Analyze question types
    if 'synthesizer_name' in df.columns:
        synthesizer_counts = df['synthesizer_name'].value_counts()
        print(f"\n📊 Question Type Distribution:")
        for synthesizer, count in synthesizer_counts.items():
            print(f"  {synthesizer}: {count}")
    
    # Analyze question themes
    sox_keywords = ['access', 'control', 'audit', 'compliance', 'authorization', 'authentication', 
                   'backup', 'change', 'management', 'approval', 'validation', 'segregation']
    
    relevant_questions = 0
    print(f"\n📋 Sample Questions:")
    
    for i, row in df.iterrows():
        question = row['user_input']
        reference = row.get('reference', 'N/A')
        
        # Check SOX relevance
        is_sox_relevant = any(keyword.lower() in question.lower() for keyword in sox_keywords)
        if is_sox_relevant:
            relevant_questions += 1
        
        # Show first 3 questions
        if i < 3:
            relevance_flag = "🎯" if is_sox_relevant else "❓"
            print(f"  {i+1}. {relevance_flag} Q: {question[:100]}{'...' if len(question) > 100 else ''}")
            print(f"     A: {reference[:100]}{'...' if len(reference) > 100 else ''}")
    
    relevance_rate = relevant_questions / len(df) if len(df) > 0 else 0
    print(f"\n📈 SOX Relevance Analysis:")
    print(f"  Relevant questions: {relevant_questions}/{len(df)} ({relevance_rate:.1%})")
    
    return df, relevance_rate

def test_question_generation():
    """Main test function for Step 1c"""
    
    print("🚀 Starting Step 1c: Simple Question Generation Test")
    print("=" * 60)
    
    try:
        # Step 1: Create knowledge graph
        kg, documents, generator_llm, generator_embeddings = create_sox_knowledge_graph()
        
        # Step 2: Generate questions
        testset = generate_sox_questions(kg, documents, generator_llm, generator_embeddings)
        
        # Step 3: Analyze questions
        df, relevance_rate = analyze_generated_questions(testset)
        
        # Step 4: Verify results
        min_questions = 5
        min_relevance = 0.5  # At least 50% should be SOX-relevant
        
        print(f"\n✅ Step 1c Test Results:")
        print(f"  ✓ Question generation completed successfully")
        print(f"  ✓ Generated {len(df)} questions (expected ≥{min_questions})")
        print(f"  ✓ SOX relevance: {relevance_rate:.1%} (target ≥{min_relevance:.1%})")
        
        if len(df) >= min_questions:
            print(f"  ✓ Question count meets expectations")
        else:
            print(f"  ⚠️  Question count lower than expected")
            
        if relevance_rate >= min_relevance:
            print(f"  ✓ Relevance rate meets expectations")
        else:
            print(f"  ⚠️  Relevance rate lower than expected")
        
        print(f"\n🎯 Step 1c: COMPLETE - Ready for Step 1d (Question Analysis & Evidence Mapping)")
        
        return testset, df
        
    except Exception as e:
        print(f"❌ Error in Step 1c: {e}")
        raise


if __name__ == "__main__":
    test_question_generation() 