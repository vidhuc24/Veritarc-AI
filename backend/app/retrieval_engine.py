"""
Core Retrieval Engine for Veritarc AI
Advanced document retrieval with metadata filtering, hybrid search, and query expansion
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class RetrievalResult:
    """Structured result from retrieval operations"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    retrieval_method: str

class VeritarcRetrievalEngine:
    def __init__(self, vector_store_path: str = "./data/chroma_db"):
        """Initialize the retrieval engine with vector store connection"""
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.openai_client = OpenAI()
        
        # Get collections
        self.controls_collection = self.client.get_collection("sox_controls")
        self.evidence_collection = self.client.get_collection("evidence_documents")
        
        # Query expansion mappings for compliance terminology
        self.compliance_expansions = {
            "access": ["authentication", "authorization", "user management", "permissions", "login"],
            "change": ["modification", "update", "deployment", "implementation", "rollback"],
            "backup": ["recovery", "restoration", "disaster", "continuity", "data protection"],
            "review": ["audit", "assessment", "evaluation", "certification", "validation"],
            "approval": ["authorization", "signature", "endorsement", "consent", "permission"],
            "testing": ["validation", "verification", "quality assurance", "QA", "test results"],
            "monitoring": ["surveillance", "tracking", "logging", "alerting", "oversight"],
            "security": ["protection", "safeguard", "control", "defense", "risk mitigation"]
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with compliance terminology synonyms"""
        expanded_queries = [query]
        
        # Add compliance terminology expansions
        for term, expansions in self.compliance_expansions.items():
            if term.lower() in query.lower():
                for expansion in expansions:
                    expanded_query = query.replace(term, expansion)
                    if expanded_query != query:
                        expanded_queries.append(expanded_query)
        
        # Add common SOX terminology
        sox_terms = ["SOX", "Sarbanes-Oxley", "ITGC", "internal control", "compliance"]
        for term in sox_terms:
            if term.lower() in query.lower():
                expanded_queries.append(query + " compliance audit")
                break
        
        return list(set(expanded_queries))  # Remove duplicates
    
    def semantic_search(self, query: str, collection, n_results: int = 5, 
                       metadata_filter: Optional[Dict] = None) -> List[RetrievalResult]:
        """Perform semantic search with optional metadata filtering"""
        
        # Build where clause for metadata filtering
        where_clause = None
        if metadata_filter:
            where_clause = {}
            for key, value in metadata_filter.items():
                if isinstance(value, list):
                    where_clause[key] = {"$in": value}
                else:
                    where_clause[key] = value
        
        # Perform search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        # Convert to RetrievalResult objects
        retrieval_results = []
        for i in range(len(results['ids'][0])):
            result = RetrievalResult(
                document_id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                similarity_score=results['distances'][0][i],
                retrieval_method="semantic_search"
            )
            retrieval_results.append(result)
        
        return retrieval_results
    
    def keyword_search(self, query: str, collection, n_results: int = 5,
                      metadata_filter: Optional[Dict] = None) -> List[RetrievalResult]:
        """Perform keyword-based search using query expansion"""
        
        # Expand query for better keyword matching
        expanded_queries = self.expand_query(query)
        
        all_results = []
        for expanded_query in expanded_queries:
            results = self.semantic_search(expanded_query, collection, n_results, metadata_filter)
            all_results.extend(results)
        
        # Remove duplicates and sort by similarity
        unique_results = {}
        for result in all_results:
            if result.document_id not in unique_results:
                unique_results[result.document_id] = result
            else:
                # Keep the better score
                if result.similarity_score < unique_results[result.document_id].similarity_score:
                    unique_results[result.document_id] = result
        
        # Sort by similarity score and return top results
        sorted_results = sorted(unique_results.values(), key=lambda x: x.similarity_score)
        return sorted_results[:n_results]
    
    def hybrid_search(self, query: str, collection, n_results: int = 5,
                     metadata_filter: Optional[Dict] = None,
                     semantic_weight: float = 0.7) -> List[RetrievalResult]:
        """Combine semantic and keyword search for better results"""
        
        # Get semantic search results
        semantic_results = self.semantic_search(query, collection, n_results * 2, metadata_filter)
        
        # Get keyword search results
        keyword_results = self.keyword_search(query, collection, n_results * 2, metadata_filter)
        
        # Combine and score results
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            combined_results[result.document_id] = {
                'result': result,
                'semantic_score': 1 - result.similarity_score,  # Convert distance to similarity
                'keyword_score': 0
            }
        
        # Add keyword results
        for result in keyword_results:
            if result.document_id in combined_results:
                combined_results[result.document_id]['keyword_score'] = 1 - result.similarity_score
            else:
                combined_results[result.document_id] = {
                    'result': result,
                    'semantic_score': 0,
                    'keyword_score': 1 - result.similarity_score
                }
        
        # Calculate hybrid scores
        hybrid_results = []
        for doc_id, scores in combined_results.items():
            hybrid_score = (semantic_weight * scores['semantic_score'] + 
                          (1 - semantic_weight) * scores['keyword_score'])
            
            result = scores['result']
            result.similarity_score = 1 - hybrid_score  # Convert back to distance
            result.retrieval_method = "hybrid_search"
            hybrid_results.append(result)
        
        # Sort by hybrid score and return top results
        hybrid_results.sort(key=lambda x: x.similarity_score)
        return hybrid_results[:n_results]
    
    def retrieve_controls_for_evidence(self, evidence_content: str, 
                                     category_filter: Optional[str] = None) -> List[RetrievalResult]:
        """Retrieve relevant SOX controls for a given evidence document"""
        
        # Extract key terms from evidence content
        key_terms = self.extract_key_terms(evidence_content)
        
        # Build query from key terms
        query = " ".join(key_terms[:5])  # Use top 5 terms
        
        # Apply category filter if specified
        metadata_filter = None
        if category_filter:
            metadata_filter = {"category": category_filter}
        
        # Use hybrid search for better control matching
        results = self.hybrid_search(query, self.controls_collection, n_results=3, 
                                   metadata_filter=metadata_filter)
        
        return results
    
    def retrieve_evidence_for_control(self, control_id: str, 
                                    document_type: Optional[str] = None,
                                    quality_level: Optional[str] = None) -> List[RetrievalResult]:
        """Retrieve relevant evidence documents for a given SOX control"""
        
        # Get control details to build query
        control_details = self.get_control_details(control_id)
        if not control_details:
            return []
        
        # Build query from control information - use available fields
        query_parts = []
        if 'name' in control_details:
            query_parts.append(control_details['name'])
        
        # Get the full control document content for better query building
        control_docs = self.controls_collection.get(ids=[control_id])
        if control_docs['documents']:
            # Extract key terms from the full control document
            control_content = control_docs['documents'][0]
            key_terms = self.extract_key_terms(control_content)
            query_parts.extend(key_terms[:3])  # Add top 3 key terms
        
        query = " ".join(query_parts)
        
        # Apply filters
        metadata_filter = {"control_id": control_id}
        if document_type:
            metadata_filter["document_type"] = document_type
        if quality_level:
            metadata_filter["quality_level"] = quality_level
        
        # Use hybrid search for better evidence matching
        results = self.hybrid_search(query, self.evidence_collection, n_results=5,
                                   metadata_filter=metadata_filter)
        
        return results
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for query building"""
        # Simple term extraction - could be enhanced with NLP
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency and return most common terms
        term_counts = {}
        for term in key_terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        return [term for term, count in sorted_terms[:10]]
    
    def get_control_details(self, control_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific SOX control"""
        results = self.controls_collection.get(ids=[control_id])
        if results['ids']:
            return results['metadatas'][0]
        return None
    
    def test_retrieval_accuracy(self):
        """Test retrieval accuracy with sample queries"""
        print("\n🧪 Testing retrieval accuracy...")
        
        # Test 1: Control retrieval for evidence
        print("Test 1: Retrieving controls for access review evidence...")
        sample_evidence = "access review report user authentication manager approval quarterly review"
        controls = self.retrieve_controls_for_evidence(sample_evidence)
        print(f"Found {len(controls)} relevant controls:")
        for i, control in enumerate(controls):
            print(f"  {i+1}. {control.metadata.get('name', 'Unknown')} (Score: {1-control.similarity_score:.3f})")
        
        # Test 2: Evidence retrieval for control
        print("\nTest 2: Retrieving evidence for SOX-ITGC-AC-02...")
        evidence = self.retrieve_evidence_for_control("SOX-ITGC-AC-02")
        print(f"Found {len(evidence)} relevant evidence documents:")
        for i, doc in enumerate(evidence):
            print(f"  {i+1}. {doc.metadata.get('document_type', 'Unknown')} - {doc.metadata.get('company', 'Unknown')} (Score: {1-doc.similarity_score:.3f})")
        
        # Test 3: Hybrid search with metadata filtering
        print("\nTest 3: Hybrid search with quality filter...")
        results = self.hybrid_search(
            "change management approval testing",
            self.evidence_collection,
            metadata_filter={"quality_level": "high"}
        )
        print(f"Found {len(results)} high-quality change management evidence:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.metadata.get('document_type', 'Unknown')} - {result.metadata.get('company', 'Unknown')} (Score: {1-result.similarity_score:.3f})")
        
        print("✅ Retrieval accuracy tests completed!")


def main():
    """Test the retrieval engine"""
    print("🚀 Testing Veritarc AI Retrieval Engine...")
    
    # Initialize retrieval engine
    engine = VeritarcRetrievalEngine()
    
    # Test retrieval accuracy
    engine.test_retrieval_accuracy()
    
    print("\n🎉 Step 2 Complete: Core Retrieval Engine Successful!")


if __name__ == "__main__":
    main() 