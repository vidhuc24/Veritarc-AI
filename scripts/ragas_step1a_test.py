"""
Step 1a: RAGAS Setup & SOX Document Loading Test
Load our SOX control documents into RAGAS format and verify proper loading
"""

import json
import os
from typing import List
from langchain.schema import Document
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_sox_controls_as_documents() -> List[Document]:
    """Load SOX controls from JSON and convert to LangChain Documents"""
    
    print("📄 Loading SOX control documents...")
    
    # Load our SOX controls
    sox_file = "data/sample_sox_controls.json"
    if not os.path.exists(sox_file):
        raise FileNotFoundError(f"SOX controls file not found: {sox_file}")
    
    with open(sox_file, 'r') as f:
        sox_controls = json.load(f)
    
    print(f"✅ Found {len(sox_controls)} SOX controls")
    
    # Convert to LangChain Documents format (required by RAGAS)
    documents = []
    for control in sox_controls:
        # Create comprehensive document content
        content = f"""Control ID: {control['control_id']}
Control Name: {control['name']}
Category: {control['category']}

Description:
{control['description']}

Validation Criteria:
{chr(10).join(f"- {criteria}" for criteria in control['validation_criteria'])}

Keywords: {', '.join(control['keywords'])}
"""
        
        # Create Document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "control_id": control['control_id'],
                "name": control['name'],
                "category": control['category'],
                "document_type": "sox_control",
                "source": "sox_controls_database"
            }
        )
        documents.append(doc)
    
    return documents

def setup_ragas_components():
    """Setup RAGAS LLM and embedding components"""
    
    print("🔧 Setting up RAGAS components...")
    
    # Initialize LLM for RAGAS (using GPT-4 as in bootcamp example)
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
    
    # Initialize embedding model for RAGAS
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    print("✅ RAGAS components initialized")
    return generator_llm, generator_embeddings

def test_document_loading():
    """Main test function for Step 1a"""
    
    print("🚀 Starting Step 1a: RAGAS Setup & SOX Document Loading Test")
    print("=" * 60)
    
    try:
        # Test 1: Load SOX documents
        documents = load_sox_controls_as_documents()
        
        # Test 2: Verify document structure
        print(f"\n📊 Document Loading Results:")
        print(f"  Total documents: {len(documents)}")
        
        # Check categories
        categories = set(doc.metadata['category'] for doc in documents)
        print(f"  Categories found: {', '.join(sorted(categories))}")
        
        # Check control IDs
        control_ids = [doc.metadata['control_id'] for doc in documents]
        print(f"  Control IDs: {', '.join(sorted(control_ids))}")
        
        # Test 3: Setup RAGAS components
        generator_llm, generator_embeddings = setup_ragas_components()
        
        # Test 4: Inspect sample document
        print(f"\n📋 Sample Document Preview:")
        print("-" * 40)
        sample_doc = documents[0]
        print(f"Control ID: {sample_doc.metadata['control_id']}")
        print(f"Category: {sample_doc.metadata['category']}")
        print(f"Content length: {len(sample_doc.page_content)} characters")
        print(f"First 200 chars: {sample_doc.page_content[:200]}...")
        
        # Test 5: Verify expected counts
        expected_controls = 9  # We should have 9 controls
        expected_categories = 3  # 3 categories
        
        assert len(documents) == expected_controls, f"Expected {expected_controls} controls, got {len(documents)}"
        assert len(categories) == expected_categories, f"Expected {expected_categories} categories, got {len(categories)}"
        
        print(f"\n✅ Step 1a Test Results:")
        print(f"  ✓ Successfully loaded {len(documents)} SOX control documents")
        print(f"  ✓ All {len(categories)} categories present: {', '.join(sorted(categories))}")
        print(f"  ✓ RAGAS components initialized successfully")
        print(f"  ✓ Document format compatible with RAGAS")
        
        print(f"\n🎯 Step 1a: COMPLETE - Ready for Step 1b (Knowledge Graph Creation)")
        
        return documents, generator_llm, generator_embeddings
        
    except Exception as e:
        print(f"❌ Error in Step 1a: {e}")
        raise


if __name__ == "__main__":
    test_document_loading() 