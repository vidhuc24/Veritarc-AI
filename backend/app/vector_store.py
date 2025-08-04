"""
Vector Store Setup for Veritarc AI
Handles loading SOX controls and evidence documents into Qdrant
"""

import json
import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VeritarcVectorStore:
    def __init__(self, persist_directory: str = "./data/qdrant_db"):
        """Initialize the vector store with Qdrant"""
        self.persist_directory = persist_directory
        
        # Initialize Qdrant client (local/embedded mode)
        self.client = QdrantClient(path=persist_directory)
        
        # Initialize OpenAI client for embeddings
        self.openai_client = OpenAI()
        
        # Collection names
        self.controls_collection = "sox_controls"
        self.evidence_collection = "evidence_documents"
        
        # Embedding model configuration
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536  # text-embedding-3-small dimensions
        
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI text-embedding-3-small"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            raise
        
    def setup_collections(self):
        """Create or recreate collections for controls and evidence"""
        print(f"Setting up Qdrant collections with {self.embedding_model}...")
        
        # Create vector configuration
        vector_config = VectorParams(
            size=self.embedding_dimension,
            distance=Distance.COSINE
        )
        
        # Create or recreate SOX controls collection
        if self.client.collection_exists(self.controls_collection):
            self.client.delete_collection(self.controls_collection)
            
        self.client.create_collection(
            collection_name=self.controls_collection,
            vectors_config=vector_config
        )
        
        # Create or recreate evidence documents collection
        if self.client.collection_exists(self.evidence_collection):
            self.client.delete_collection(self.evidence_collection)
            
        self.client.create_collection(
            collection_name=self.evidence_collection,
            vectors_config=vector_config
        )
        
        print(f"✅ Collections ready with {self.embedding_model}: {self.controls_collection}, {self.evidence_collection}")
        
    def load_sox_controls(self, controls_file: str = "data/sample_sox_controls.json"):
        """Load SOX controls into the vector store"""
        print(f"Loading SOX controls from {controls_file}...")
        
        if not os.path.exists(controls_file):
            raise FileNotFoundError(f"Controls file not found: {controls_file}")
            
        with open(controls_file, 'r') as f:
            controls = json.load(f)
        
        # Prepare points for Qdrant
        points = []
        
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
            
            # Generate embedding
            embedding = self._get_embedding(doc_text.strip())
            
            # Generate UUID for Qdrant while keeping original ID in payload
            point_uuid = str(uuid.uuid4())
            
            # Create point
            point = PointStruct(
                id=point_uuid,
                vector=embedding,
                payload={
                    "control_id": control["control_id"],  # Keep original ID for reference
                    "name": control["name"], 
                    "category": control["category"],
                    "description": control["description"],
                    "validation_criteria": control["validation_criteria"],
                    "keywords": control["keywords"],
                    "document_type": "sox_control",
                    "content": doc_text.strip()
                }
            )
            points.append(point)
        
        # Upload points to collection
        self.client.upsert(
            collection_name=self.controls_collection,
            points=points
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
        
        # Prepare points for Qdrant
        points = []
        
        for i, doc in enumerate(evidence_docs):
            # Use the document content as the main text
            doc_text = doc["content"]
            
            # Generate embedding
            embedding = self._get_embedding(doc_text)
            
            # Generate UUID for Qdrant while keeping original ID in payload
            point_uuid = str(uuid.uuid4())
            
            # Create point
            point = PointStruct(
                id=point_uuid,
                vector=embedding,
                payload={
                    "evidence_id": f"evidence_{i}_{doc['document_type']}",  # Keep original ID for reference
                    "document_type": doc["document_type"],
                    "control_id": doc["control_id"],
                    "company": doc["company"],
                    "quality_level": doc["quality_level"],
                    "expected_result": doc["expected_result"],
                    "generated_date": doc.get("generated_date", ""),
                    "content": doc_text
                }
            )
            points.append(point)
        
        # Upload points to collection
        self.client.upsert(
            collection_name=self.evidence_collection,
            points=points
        )
        
        print(f"✅ Loaded {len(evidence_docs)} evidence documents into vector store")
        return len(evidence_docs)
    
    def search_controls(self, query: str, limit: int = 5, score_threshold: float = 0.0, 
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search SOX controls using semantic similarity"""
        query_embedding = self._get_embedding(query)
        
        # Convert filters to Qdrant filter format if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        search_result = self.client.query_points(
            collection_name=self.controls_collection,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=limit,
            score_threshold=score_threshold
        )
        
        results = []
        for point in search_result.points:
            results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload
            })
        
        return results
    
    def search_evidence(self, query: str, limit: int = 5, score_threshold: float = 0.0, 
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search evidence documents using semantic similarity"""
        query_embedding = self._get_embedding(query)
        
        # Convert filters to Qdrant filter format if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        search_result = self.client.query_points(
            collection_name=self.evidence_collection,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=limit,
            score_threshold=score_threshold
        )
        
        results = []
        for point in search_result.points:
            results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload
            })
        
        return results
    
    def get_by_id(self, collection: str, point_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific point by ID"""
        try:
            points = self.client.retrieve(
                collection_name=collection,
                ids=[point_id]
            )
            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "payload": point.payload
                }
            return None
        except Exception as e:
            print(f"❌ Error retrieving point {point_id}: {e}")
            return None
    
    def get_control_by_control_id(self, control_id: str) -> Optional[Dict[str, Any]]:
        """Find a control by its original control_id"""
        try:
            search_result = self.client.scroll(
                collection_name=self.controls_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="control_id",
                            match=MatchValue(value=control_id)
                        )
                    ]
                ),
                limit=1
            )
            
            if search_result[0]:  # search_result is (points, next_page_offset)
                point = search_result[0][0]
                return {
                    "id": point.id,
                    "payload": point.payload
                }
            return None
        except Exception as e:
            print(f"❌ Error retrieving control {control_id}: {e}")
            return None
    
    def get_evidence_by_evidence_id(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Find evidence by its original evidence_id"""
        try:
            search_result = self.client.scroll(
                collection_name=self.evidence_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="evidence_id",
                            match=MatchValue(value=evidence_id)
                        )
                    ]
                ),
                limit=1
            )
            
            if search_result[0]:  # search_result is (points, next_page_offset)
                point = search_result[0][0]
                return {
                    "id": point.id,
                    "payload": point.payload
                }
            return None
        except Exception as e:
            print(f"❌ Error retrieving evidence {evidence_id}: {e}")
            return None
    
    def test_basic_search(self):
        """Test basic similarity search functionality"""
        print("\n🧪 Testing basic search functionality...")
        
        # Test 1: Search for access control related content
        print("Test 1: Searching for 'access control' in controls...")
        results = self.search_controls("access control user authentication", limit=3)
        
        print(f"Found {len(results)} relevant controls:")
        for i, result in enumerate(results):
            content = result['payload']['content'][:100]
            print(f"  {i+1}. Score: {result['score']:.3f} - {content}...")
        
        # Test 2: Search for evidence documents
        print("\nTest 2: Searching for 'access review' in evidence...")
        results = self.search_evidence("access review report user access", limit=2)
        
        print(f"Found {len(results)} relevant evidence documents:")
        for i, result in enumerate(results):
            content = result['payload']['content'][:100]
            print(f"  {i+1}. Score: {result['score']:.3f} - {content}...")
        
        # Test 3: Check collection statistics
        print(f"\nTest 3: Collection statistics:")
        controls_info = self.client.get_collection(self.controls_collection)
        evidence_info = self.client.get_collection(self.evidence_collection)
        
        print(f"  Controls collection: {controls_info.points_count} documents")
        print(f"  Evidence collection: {evidence_info.points_count} documents")
        
        print("✅ Basic search tests completed successfully!")
    
    def get_collection_info(self):
        """Get information about the loaded collections"""
        print("\n📊 Vector Store Information:")
        
        controls_info = self.client.get_collection(self.controls_collection)
        evidence_info = self.client.get_collection(self.evidence_collection)
        
        print(f"  Controls Collection: {controls_info.points_count} documents")
        print(f"  Evidence Collection: {evidence_info.points_count} documents")
        print(f"  Persist Directory: {self.persist_directory}")
        print(f"  Embedding Model: {self.embedding_model}")
        print(f"  Vector Dimensions: {self.embedding_dimension}")
        
        # Sample some documents
        if controls_info.points_count > 0:
            sample_controls = self.client.scroll(
                collection_name=self.controls_collection,
                limit=2
            )
            control_ids = [point.payload.get("control_id", point.id) for point in sample_controls[0]]
            print(f"\nSample Control IDs: {control_ids}")
        
        if evidence_info.points_count > 0:
            sample_evidence = self.client.scroll(
                collection_name=self.evidence_collection,
                limit=2
            )
            evidence_ids = [point.payload.get("evidence_id", point.id) for point in sample_evidence[0]]
            print(f"Sample Evidence IDs: {evidence_ids}")


def main():
    """Main function to set up and test the vector store"""
    print("🚀 Setting up Veritarc AI Vector Store with Qdrant...")
    
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
        
        print("\n🎉 Step 1 Complete: Qdrant Vector Store Setup Successful!")
        
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        raise


if __name__ == "__main__":
    main() 