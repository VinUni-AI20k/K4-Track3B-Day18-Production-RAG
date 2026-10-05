"""Prepare M5 cache independently of baseline evaluation."""
from src.m1_chunking import load_documents, chunk_hierarchical
from src.m5_enrichment import enrich_chunks


def main():
    chunks = []
    for doc in load_documents():
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_texts = {p.metadata["parent_id"]: p.text for p in parents}
        for child in children:
            chunks.append({"text": child.text, "metadata": {**child.metadata,
                "parent_id": child.parent_id, "parent_text": parent_texts[child.parent_id]}})
    enriched = enrich_chunks(chunks, cache_dir=".run_cache/enrichment")
    count = sum(c.auto_metadata.get("enrichment_mode") == "llm" for c in enriched)
    print(f"Prepared M5 cache: {count}/{len(enriched)} AI-enriched chunks", flush=True)


if __name__ == "__main__":
    main()
