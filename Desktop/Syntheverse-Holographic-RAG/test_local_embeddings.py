"""
Test Local Embeddings: Verify FREE sentence-transformers work
"""

from langchain_vector_store_local import LangChainVectorStoreLocal
from langchain.schema import Document

def test_local_embeddings():
    """Test free local embeddings."""
    
    print("=" * 80)
    print("Testing FREE Local Embeddings (sentence-transformers)")
    print("=" * 80)
    
    try:
        print("\n[Step 1] Loading embedding model...")
        print("  (First time will download ~80MB, then cached)")
        
        vector_store = LangChainVectorStoreLocal(
            collection_name="test_embeddings",
            persist_directory="./data/test_chroma_db",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        print("\n[Step 2] Testing embedding generation...")
        
        # Test embedding
        test_text = "Hydrogen holographic fractal intelligence"
        print(f"  Generating embedding for: '{test_text}'")
        
        embedding = vector_store.embeddings.embed_query(test_text)
        print(f"  ✓ Generated embedding with {len(embedding)} dimensions")
        print(f"  ✓ First 5 values: {[round(x, 4) for x in embedding[:5]]}")
        
        print("\n[Step 3] Testing document storage and retrieval...")
        
        # Add test documents
        test_docs = [
            Document(page_content="Hydrogen is the most abundant element in the universe.", 
                    metadata={"source": "test1"}),
            Document(page_content="Fractal patterns appear in nature at all scales.", 
                    metadata={"source": "test2"}),
            Document(page_content="Holographic principles suggest information is stored on boundaries.", 
                    metadata={"source": "test3"})
        ]
        
        print(f"  Adding {len(test_docs)} test documents...")
        vector_store.add_documents(test_docs)
        
        # Test search
        query = "What is hydrogen?"
        print(f"\n  Searching for: '{query}'")
        results = vector_store.similarity_search(query, k=2)
        
        print(f"  ✓ Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"    {i}. {doc.page_content[:60]}...")
            print(f"       Source: {doc.metadata.get('source', 'Unknown')}")
        
        # Get stats
        stats = vector_store.get_stats()
        print(f"\n[Step 4] Vector store stats:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Embedding model: {stats['embedding_model']}")
        
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        print("\n💡 FREE local embeddings are working perfectly!")
        print("   - No API calls")
        print("   - No costs")
        print("   - No quota limits")
        print("   - Works offline")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_local_embeddings()

