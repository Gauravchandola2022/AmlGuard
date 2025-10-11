"""
AML 360 Vector Database Client
Handles embeddings and RAG retrieval for transaction explanations
"""

import chromadb
from chromadb.config import Settings
import hashlib
import json
from typing import List, Dict, Any, Optional
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
import logging
from pathlib import Path
import os

from utils.logging_config import setup_logging

logger = setup_logging(__name__)

class AMLVectorStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """Initialize Chroma vector database"""
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("aml_transactions")
        except:
            self.collection = self.client.create_collection(
                name="aml_transactions",
                metadata={"description": "AML flagged transactions for RAG retrieval"}
            )
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model: all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.embedding_model = None
        else:
            logger.warning("sentence-transformers not available - embeddings functionality disabled")
            self.embedding_model = None
    
    def create_document_text(self, transaction_data: Dict[str, Any]) -> str:
        """Create document text for embedding from transaction data"""
        
        # Extract key information
        txn_id = transaction_data.get('transaction_id', 'unknown')
        amount_usd = transaction_data.get('amount_usd', 0)
        score_breakdown = transaction_data.get('score_breakdown', [])
        payment_instruction = transaction_data.get('payment_instruction', '')
        account_id = transaction_data.get('account_id') or transaction_data.get('account_key', '')
        originator_country = transaction_data.get('originator_country', '')
        beneficiary_country = transaction_data.get('beneficiary_country', '')
        transaction_date = transaction_data.get('transaction_date', '')
        
        # Format document text
        doc_text = f"""
TXN_ID: {txn_id}
Amount USD: {amount_usd}
Rules: {json.dumps(score_breakdown, indent=2)}
Payment instruction: {payment_instruction}
Account ID: {account_id}
Originator country: {originator_country}
Beneficiary country: {beneficiary_country}
Transaction date: {transaction_date}
        """.strip()
        
        return doc_text
    
    def add_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """Add flagged transaction to vector store"""
        if not self.embedding_model:
            logger.warning("No embedding model available")
            return False
        
        try:
            transaction_id = transaction_data.get('transaction_id')
            if not transaction_id:
                logger.error("Transaction ID is required")
                return False
            
            # Create document text
            doc_text = self.create_document_text(transaction_data)
            
            # Generate embedding
            embedding = self.embedding_model.encode(doc_text).tolist()
            
            # Prepare metadata
            metadata = {
                'transaction_id': transaction_id,
                'account_id': transaction_data.get('account_id') or transaction_data.get('account_key', ''),
                'total_score': transaction_data.get('total_score', 0),
                'suspicious': transaction_data.get('suspicious', False),
                'amount_usd': transaction_data.get('amount_usd', 0),
                'beneficiary_country': transaction_data.get('beneficiary_country', ''),
                'originator_country': transaction_data.get('originator_country', ''),
                'timestamp': transaction_data.get('transaction_date', ''),
                'rule_hits_json': json.dumps(transaction_data.get('score_breakdown', []))
            }
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[metadata],
                ids=[transaction_id]
            )
            
            logger.debug(f"Added transaction {transaction_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding transaction to vector store: {e}")
            return False
    
    def batch_add_transactions(self, transactions: List[Dict[str, Any]]) -> int:
        """Batch add multiple transactions to vector store"""
        if not self.embedding_model:
            logger.warning("No embedding model available")
            return 0
        
        added_count = 0
        batch_size = 100  # Process in batches
        
        try:
            # Filter only suspicious transactions
            suspicious_txns = [txn for txn in transactions if txn.get('suspicious', False)]
            
            for i in range(0, len(suspicious_txns), batch_size):
                batch = suspicious_txns[i:i + batch_size]
                
                embeddings = []
                documents = []
                metadatas = []
                ids = []
                
                for txn in batch:
                    try:
                        transaction_id = txn.get('transaction_id')
                        if not transaction_id:
                            continue
                        
                        # Create document text and embedding
                        doc_text = self.create_document_text(txn)
                        embedding = self.embedding_model.encode(doc_text).tolist()
                        
                        # Prepare metadata
                        metadata = {
                            'transaction_id': transaction_id,
                            'account_id': txn.get('account_id') or txn.get('account_key', ''),
                            'total_score': txn.get('total_score', 0),
                            'suspicious': txn.get('suspicious', False),
                            'amount_usd': txn.get('amount_usd', 0),
                            'beneficiary_country': txn.get('beneficiary_country', ''),
                            'originator_country': txn.get('originator_country', ''),
                            'timestamp': txn.get('transaction_date', ''),
                            'rule_hits_json': json.dumps(txn.get('score_breakdown', []))
                        }
                        
                        embeddings.append(embedding)
                        documents.append(doc_text)
                        metadatas.append(metadata)
                        ids.append(transaction_id)
                        
                    except Exception as e:
                        logger.warning(f"Error processing transaction {txn.get('transaction_id')}: {e}")
                        continue
                
                # Add batch to collection
                if ids:
                    self.collection.add(
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    added_count += len(ids)
            
            logger.info(f"Batch added {added_count} transactions to vector store")
            
        except Exception as e:
            logger.error(f"Error in batch add: {e}")
        
        return added_count
    
    def retrieve_evidence(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar transactions for RAG"""
        if not self.embedding_model:
            logger.warning("No embedding model available for retrieval")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # Search similar documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            evidence = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    evidence.append({
                        'document': doc,
                        'metadata': metadata,
                        'similarity': 1 - distance,  # Convert distance to similarity
                        'transaction_id': metadata.get('transaction_id'),
                        'total_score': metadata.get('total_score'),
                        'rule_hits_json': metadata.get('rule_hits_json')
                    })
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error retrieving evidence: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand the data
            sample_results = self.collection.query(
                query_embeddings=[[0.0] * 384],  # Dummy query
                n_results=min(10, count),
                include=['metadatas']
            ) if count > 0 else {'metadatas': [[]]}
            
            # Analyze countries and scores
            countries = {}
            scores = []
            
            if sample_results['metadatas'] and sample_results['metadatas'][0]:
                for metadata in sample_results['metadatas'][0]:
                    country = metadata.get('beneficiary_country', 'Unknown')
                    countries[country] = countries.get(country, 0) + 1
                    scores.append(metadata.get('total_score', 0))
            
            return {
                'total_documents': count,
                'top_countries': countries,
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'min_score': min(scores) if scores else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_documents': 0}
    
    def delete_transaction(self, transaction_id: str) -> bool:
        """Delete transaction from vector store"""
        try:
            self.collection.delete(ids=[transaction_id])
            logger.debug(f"Deleted transaction {transaction_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting transaction: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            # Delete collection and recreate
            self.client.delete_collection("aml_transactions")
            self.collection = self.client.create_collection(
                name="aml_transactions",
                metadata={"description": "AML flagged transactions for RAG retrieval"}
            )
            logger.info("Cleared vector store collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

class RAGChatbot:
    def __init__(self, vector_store: AMLVectorStore):
        self.vector_store = vector_store
        
        # Try to initialize LLM (fallback to deterministic responses)
        self.llm_available = False
        try:
            # Try to use a simple LLM or API
            # For now, we'll use deterministic responses
            pass
        except Exception as e:
            logger.info("LLM not available, using deterministic responses")
    
    def chat(self, query: str, transaction_id: Optional[str] = None) -> str:
        """Generate response using RAG"""
        
        # Retrieve relevant evidence
        if transaction_id:
            # Focus query on specific transaction
            query = f"Transaction {transaction_id}: {query}"
        
        evidence = self.vector_store.retrieve_evidence(query, top_k=3)
        
        if not evidence:
            return self._fallback_response(query, transaction_id)
        
        # Generate response based on evidence
        return self._generate_response(query, evidence, transaction_id)
    
    def _generate_response(self, query: str, evidence: List[Dict[str, Any]], transaction_id: Optional[str] = None) -> str:
        """Generate response based on retrieved evidence"""
        
        if self.llm_available:
            # Use LLM to generate response
            return self._llm_response(query, evidence, transaction_id)
        else:
            # Use deterministic response generation
            return self._deterministic_response(query, evidence, transaction_id)
    
    def _deterministic_response(self, query: str, evidence: List[Dict[str, Any]], transaction_id: Optional[str] = None) -> str:
        """Generate deterministic response based on evidence"""
        
        response_parts = []
        
        if transaction_id:
            # Find specific transaction in evidence
            target_txn = None
            for item in evidence:
                if item.get('transaction_id') == transaction_id:
                    target_txn = item
                    break
            
            if target_txn:
                metadata = target_txn['metadata']
                rules_json = metadata.get('rule_hits_json', '[]')
                
                try:
                    rules = json.loads(rules_json)
                    rule_names = [rule.get('rule_name', 'Unknown') for rule in rules]
                    
                    response_parts.append(f"Transaction {transaction_id} was flagged with a total score of {metadata.get('total_score', 0)}.")
                    
                    if rule_names:
                        response_parts.append(f"The following rules triggered: {', '.join(rule_names)}.")
                    
                    # Specific rule explanations
                    for rule in rules:
                        rule_name = rule.get('rule_name', '')
                        evidence_text = rule.get('evidence', '')
                        
                        if rule_name == 'BeneficiaryHighRisk':
                            response_parts.append(f"The beneficiary country was flagged as high-risk: {evidence_text}")
                        elif rule_name == 'SuspiciousKeyword':
                            response_parts.append(f"Suspicious keywords were detected: {evidence_text}")
                        elif rule_name == 'LargeAmount':
                            response_parts.append(f"Large amount threshold exceeded: {evidence_text}")
                        elif rule_name == 'Structuring':
                            response_parts.append(f"Potential structuring pattern detected: {evidence_text}")
                        elif rule_name == 'RoundedAmounts':
                            response_parts.append(f"Rounded amount pattern detected: {evidence_text}")
                    
                    # Recommendations
                    score = metadata.get('total_score', 0)
                    if score >= 10:
                        response_parts.append("\nRecommended actions: Immediate investigation required. Consider filing SAR.")
                    elif score >= 6:
                        response_parts.append("\nRecommended actions: Enhanced due diligence. Review customer profile.")
                    else:
                        response_parts.append("\nRecommended actions: Monitor for patterns. Document findings.")
                    
                except Exception as e:
                    response_parts.append(f"Transaction {transaction_id} was flagged but detailed analysis is unavailable.")
            
            else:
                response_parts.append(f"Transaction {transaction_id} was not found in the flagged transactions database.")
        
        else:
            # General query about patterns
            total_evidence = len(evidence)
            if total_evidence > 0:
                response_parts.append(f"Found {total_evidence} similar flagged transactions.")
                
                # Analyze common patterns
                countries = {}
                rules = {}
                total_score = 0
                
                for item in evidence:
                    metadata = item['metadata']
                    country = metadata.get('beneficiary_country', 'Unknown')
                    countries[country] = countries.get(country, 0) + 1
                    total_score += metadata.get('total_score', 0)
                    
                    rules_json = metadata.get('rule_hits_json', '[]')
                    try:
                        rule_list = json.loads(rules_json)
                        for rule in rule_list:
                            rule_name = rule.get('rule_name', 'Unknown')
                            rules[rule_name] = rules.get(rule_name, 0) + 1
                    except:
                        pass
                
                # Most common patterns
                if countries:
                    top_country = max(countries, key=countries.get)
                    response_parts.append(f"Most common beneficiary country: {top_country} ({countries[top_country]} transactions).")
                
                if rules:
                    top_rule = max(rules, key=rules.get)
                    response_parts.append(f"Most triggered rule: {top_rule} ({rules[top_rule]} times).")
                
                avg_score = total_score / total_evidence if total_evidence > 0 else 0
                response_parts.append(f"Average risk score: {avg_score:.1f}")
        
        if not response_parts:
            return self._fallback_response(query, transaction_id)
        
        return " ".join(response_parts)
    
    def _fallback_response(self, query: str, transaction_id: Optional[str] = None) -> str:
        """Fallback response when no evidence is found"""
        if transaction_id:
            return f"No detailed information found for transaction {transaction_id}. Please verify the transaction ID or check if it was flagged by the system."
        else:
            return "No relevant flagged transactions found for your query. Try adjusting your search terms or check if transactions have been processed."
    
    def _llm_response(self, query: str, evidence: List[Dict[str, Any]], transaction_id: Optional[str] = None) -> str:
        """Generate LLM-based response (placeholder for future implementation)"""
        # This would integrate with actual LLM service
        return self._deterministic_response(query, evidence, transaction_id)

# Global instances
vector_store = AMLVectorStore()
chatbot = RAGChatbot(vector_store)

def get_vector_store() -> AMLVectorStore:
    """Get vector store instance"""
    return vector_store

def get_chatbot() -> RAGChatbot:
    """Get chatbot instance"""
    return chatbot
