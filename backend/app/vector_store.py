"""
Vector Store Setup for Veritarc AI
Handles loading SOX controls and evidence documents into Chroma
"""

import json
import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VeritarcVectorStore:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """Initialize the vector store with Chroma"""
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.openai_client = OpenAI()
        
        # Initialize collections
        self.controls_collection = None
        self.evidence_collection = None
        
    def setup_collections(self):
        """Create or get existing collections for controls and evidence"""
        print("Setting up Chroma collections...")
        
        # Collection for SOX controls
        self.controls_collection = self.client.get_or_create_collection(
            name="sox_controls",
            metadata={"description": "SOX ITGC control requirements and validation criteria"}
        )
        
        # Collection for evidence documents
        self.evidence_collection = self.client.get_or_create_collection(
            name="evidence_documents", 
            metadata={"description": "Synthetic audit evidence documents for testing"}
        )
        
        print(f"✅ Collections ready: {self.controls_collection.name}, {self.evidence_collection.name}")
        
    def load_sox_controls(self, controls_file: str = "data/sample_sox_controls.json"):
        """Load SOX controls into the vector store"""
        print(f"Loading SOX controls from {controls_file}...")
        
        if not os.path.exists(controls_file):
            raise FileNotFoundError(f"Controls file not found: {controls_file}")
            
        with open(controls_file, 'r') as f:
            controls = json.load(f)
        
        # Prepare documents for embedding
        documents = []
        metadatas = []
        ids = []
        
        for control in controls:
            # Create document text from control information
            doc_text = f"""
            Control ID: {control['control_id']}
            Name: {control['name']}
            Category: {control['category']}
            Description: {control['description']}
            Validation Criteria: {'; '.join(control['validation_criteria'])}
            Keywords: {', '.join(control['keywords'])}
            """
            
            documents.append(doc_text)
            metadatas.append({
                "control_id": control["control_id"],
                "name": control["name"], 
                "category": control["category"],
                "document_type": "sox_control"
            })
            ids.append(control["control_id"])
        
        # Add to collection
        self.controls_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Loaded {len(controls)} SOX controls into vector store")
        return len(controls)
    
    def load_evidence_documents(self, evidence_file: str = "data/synthetic_evidence/test_evidence.json"):
        """Load evidence documents into the vector store"""
        print(f"Loading evidence documents from {evidence_file}...")
        
        if not os.path.exists(evidence_file):
            raise FileNotFoundError(f"Evidence file not found: {evidence_file}")
            
        with open(evidence_file, 'r') as f:
            evidence_docs = json.load(f)
        
        # Prepare documents for embedding
        documents = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(evidence_docs):
            # Use the document content as the main text
            doc_text = doc["content"]
            
            documents.append(doc_text)
            metadatas.append({
                "document_type": doc["document_type"],
                "control_id": doc["control_id"],
                "company": doc["company"],
                "quality_level": doc["quality_level"],
                "expected_result": doc["expected_result"]
            })
            ids.append(f"evidence_{i}_{doc['document_type']}")
        
        # Add to collection
        self.evidence_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Loaded {len(evidence_docs)} evidence documents into vector store")
        return len(evidence_docs)
    
    def test_basic_search(self):
        """Test basic similarity search functionality"""
        print("\n🧪 Testing basic search functionality...")
        
        # Test 1: Search for access control related content
        print("Test 1: Searching for 'access control' in controls...")
        results = self.controls_collection.query(
            query_texts=["access control user authentication"],
            n_results=3
        )
        
        print(f"Found {len(results['documents'][0])} relevant controls:")
        for i, doc in enumerate(results['documents'][0]):
            print(f"  {i+1}. {doc[:100]}...")
        
        # Test 2: Search for evidence documents
        print("\nTest 2: Searching for 'access review' in evidence...")
        results = self.evidence_collection.query(
            query_texts=["access review report user access"],
            n_results=2
        )
        
        print(f"Found {len(results['documents'][0])} relevant evidence documents:")
        for i, doc in enumerate(results['documents'][0]):
            print(f"  {i+1}. {doc[:100]}...")
        
        # Test 3: Check collection statistics
        print(f"\nTest 3: Collection statistics:")
        print(f"  Controls collection: {self.controls_collection.count()} documents")
        print(f"  Evidence collection: {self.evidence_collection.count()} documents")
        
        print("✅ Basic search tests completed successfully!")
    
    def get_collection_info(self):
        """Get information about the loaded collections"""
        print("\n📊 Vector Store Information:")
        print(f"  Controls Collection: {self.controls_collection.count()} documents")
        print(f"  Evidence Collection: {self.evidence_collection.count()} documents")
        print(f"  Persist Directory: {self.persist_directory}")
        
        # Sample some documents
        if self.controls_collection.count() > 0:
            sample_controls = self.controls_collection.get(limit=2)
            print(f"\nSample Control IDs: {sample_controls['ids']}")
        
        if self.evidence_collection.count() > 0:
            sample_evidence = self.evidence_collection.get(limit=2)
            print(f"Sample Evidence IDs: {sample_evidence['ids']}")


def main():
    """Main function to set up and test the vector store"""
    print("🚀 Setting up Veritarc AI Vector Store...")
    
    # Initialize vector store
    vector_store = VeritarcVectorStore()
    
    # Setup collections
    vector_store.setup_collections()
    
    # Load data
    try:
        num_controls = vector_store.load_sox_controls()
        num_evidence = vector_store.load_evidence_documents()
        
        print(f"\n✅ Data loading complete:")
        print(f"  SOX Controls: {num_controls}")
        print(f"  Evidence Documents: {num_evidence}")
        
        # Test functionality
        vector_store.test_basic_search()
        vector_store.get_collection_info()
        
        print("\n🎉 Step 1 Complete: Vector Store Setup Successful!")
        
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        raise


if __name__ == "__main__":
    main() 