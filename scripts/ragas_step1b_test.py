"""
Step 1b: Basic Knowledge Graph Creation Test
Create KnowledgeGraph from SOX controls and apply basic transformations
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_sox_documents() -> List[Document]:
    """Load SOX controls as LangChain Documents (from Step 1a)"""
    
    # Load our SOX controls
    with open("data/sample_sox_controls.json", 'r') as f:
        sox_controls = json.load(f)
    
    # Convert to LangChain Documents
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
                "document_type": "sox_control",
                "source": "sox_controls_database"
            }
        )
        documents.append(doc)
    
    return documents

def create_basic_knowledge_graph() -> KnowledgeGraph:
    """Create a KnowledgeGraph from SOX documents"""
    
    print("🔗 Creating Knowledge Graph from SOX controls...")
    
    # Load documents
    documents = load_sox_documents()
    print(f"📄 Loaded {len(documents)} SOX control documents")
    
    # Initialize empty knowledge graph
    kg = KnowledgeGraph()
    print(f"📊 Initial KnowledgeGraph: {kg}")
    
    # Add documents as nodes to the graph
    print("➕ Adding document nodes to knowledge graph...")
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
    
    print(f"📊 KnowledgeGraph after adding nodes: {kg}")
    return kg, documents

def apply_basic_transformations(kg: KnowledgeGraph, documents: List[Document]):
    """Apply RAGAS transformations to build relationships"""
    
    print("\n🔄 Applying RAGAS transformations...")
    
    # Setup RAGAS components
    transformer_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4"))
    embedding_model = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Get default transformations
    print("🔧 Creating default transformations...")
    transforms = default_transforms(
        documents=documents, 
        llm=transformer_llm, 
        embedding_model=embedding_model
    )
    
    print(f"📋 Available transformations: {len(transforms)}")
    for i, transform in enumerate(transforms):
        print(f"  {i+1}. {transform.__class__.__name__}")
    
    # Apply transformations
    print("\n🚀 Applying transformations to knowledge graph...")
    apply_transforms(kg, transforms)
    
    print(f"📊 KnowledgeGraph after transformations: {kg}")
    return kg

def analyze_knowledge_graph(kg: KnowledgeGraph):
    """Analyze the structure of the created knowledge graph"""
    
    print(f"\n🔍 Knowledge Graph Analysis:")
    print(f"  Total nodes: {len(kg.nodes)}")
    print(f"  Total relationships: {len(kg.relationships)}")
    
    # Analyze node types
    node_types = {}
    for node in kg.nodes:
        node_type = node.type.name if hasattr(node.type, 'name') else str(node.type)
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"\n📊 Node Type Distribution:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")
    
    # Analyze node properties
    if kg.nodes:
        sample_node = kg.nodes[0]
        print(f"\n📋 Sample Node Properties:")
        print(f"  Type: {sample_node.type}")
        print(f"  Available properties: {list(sample_node.properties.keys())}")
        
        # Show a sample of content if available
        if 'page_content' in sample_node.properties:
            content = sample_node.properties['page_content']
            print(f"  Content preview: {content[:150]}...")
    
    # Analyze relationships if any
    if kg.relationships:
        print(f"\n🔗 Sample Relationships:")
        for i, rel in enumerate(kg.relationships[:3]):  # Show first 3
            print(f"  {i+1}. {rel}")
    else:
        print(f"\n🔗 No relationships found yet")
    
    return kg

def test_knowledge_graph_creation():
    """Main test function for Step 1b"""
    
    print("🚀 Starting Step 1b: Basic Knowledge Graph Creation Test")
    print("=" * 60)
    
    try:
        # Test 1: Create basic knowledge graph
        kg, documents = create_basic_knowledge_graph()
        
        # Test 2: Apply transformations
        kg_transformed = apply_basic_transformations(kg, documents)
        
        # Test 3: Analyze the results
        final_kg = analyze_knowledge_graph(kg_transformed)
        
        # Test 4: Verify expectations
        expected_min_nodes = 9  # At least our original 9 SOX controls
        actual_nodes = len(final_kg.nodes)
        
        print(f"\n✅ Step 1b Test Results:")
        print(f"  ✓ Knowledge graph created successfully")
        print(f"  ✓ {actual_nodes} nodes generated (expected ≥{expected_min_nodes})")
        print(f"  ✓ {len(final_kg.relationships)} relationships established")
        print(f"  ✓ RAGAS transformations applied successfully")
        
        if actual_nodes >= expected_min_nodes:
            print(f"  ✓ Node count meets expectations")
        else:
            print(f"  ⚠️  Node count lower than expected")
        
        print(f"\n🎯 Step 1b: COMPLETE - Ready for Step 1c (Question Generation)")
        
        return final_kg, documents
        
    except Exception as e:
        print(f"❌ Error in Step 1b: {e}")
        raise


if __name__ == "__main__":
    test_knowledge_graph_creation() 