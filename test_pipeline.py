"""
End-to-end test script for FinVerify AI
Generates a realistic financial filing PDF, indexes it into FAISS,
and verifies an answerable question and an unanswerable question.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from pypdf import PdfWriter
from ingest import extract_chunks_from_pdf
from rag import VectorStoreManager, LLMGenerator, FinVerifyRAG
from guardrail import EXACT_FALLBACK_ANSWER

def create_sample_financial_pdf(filename: str = "sample_financial_report.pdf") -> str:
    from pypdf import PageObject
    
    # We can create a simple PDF with pypdf or reportlab, or if we want standard PDF text:
    # A lightweight way in pypdf:
    # We can write text content to a PDF using basic PDF structure or ReportLab/Fpdf if available,
    # or use PyPDF canvas if present, or write raw PDF stream.
    
    # Let's generate a valid minimal PDF with text streams
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >> endobj\n"
        b"4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >> endobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    )
    
    # Page 1 Text
    p1_text = (
        "Apex Financial Holdings Inc. Form 10-Q Quarterly Report for Q3 2024. "
        "Total net revenue reached $142.8 million, up 14.2% from $125.0 million in Q3 2023. "
        "Net income was $38.4 million or $1.42 per diluted share. Operating expenses were $94.6 million, "
        "driven by investments in algorithmic risk systems and compliance automation. "
        "Gross margin expanded to 66.2% reflecting operational efficiency across enterprise lending divisions."
    )
    # PDF stream format for text
    stream1 = f"BT /F1 12 Tf 50 700 Td ({p1_text}) Tj ET".encode("latin1")
    obj6 = f"6 0 obj << /Length {len(stream1)} >> stream\n".encode("latin1") + stream1 + b"\nendstream\nendobj\n"
    
    # Page 2 Text
    p2_text = (
        "Liquidity and Capital Resources. Cash and marketable securities totaled $412.5 million. "
        "Total long-term debt was $185.0 million with a weighted average interest rate of 4.85%. "
        "On October 15, 2024, the Board approved a quarterly cash dividend of $0.35 per share, "
        "payable on November 20, 2024. Capital expenditures totaled $18.2 million for Q3. "
        "The Tier 1 Common Equity Capital Ratio stood at 13.8%, exceeding regulatory minimums."
    )
    stream2 = f"BT /F1 12 Tf 50 700 Td ({p2_text}) Tj ET".encode("latin1")
    obj7 = f"7 0 obj << /Length {len(stream2)} >> stream\n".encode("latin1") + stream2 + b"\nendstream\nendobj\n"
    
    xref_offset = len(pdf_content) + len(obj6) + len(obj7)
    trailer = (
        b"xref\n0 8\n0000000000 65535 f \n"
        b"trailer << /Size 8 /Root 1 0 R >>\n"
        b"startxref\n0\n%%EOF"
    )
    
    full_pdf = pdf_content + obj6 + obj7 + trailer
    with open(filename, "wb") as f:
        f.write(full_pdf)
    return filename

def test_pipeline():
    print("=== STARTING FINVERIFY AI PIPELINE TEST ===")
    pdf_path = create_sample_financial_pdf("Apex_Q3_Report.pdf")
    print(f"Created test PDF: {pdf_path}")
    
    # 1. Ingestion & Chunking
    chunks = extract_chunks_from_pdf(pdf_path, document_name="Apex_Q3_Report.pdf", chunk_size=700, chunk_overlap=120)
    print(f"Extracted {len(chunks)} chunks:")
    for c in chunks:
        print(f"  - ID: {c.chunk_id} | Doc: {c.document_name} | Page: {c.page_number} | Len: {len(c.content)}")
        assert c.chunk_id, "chunk_id must not be empty"
        assert c.document_name == "Apex_Q3_Report.pdf", "document_name preserved"
        assert c.page_number in [1, 2], "page_number preserved"
    
    # 2. Vector Store & FAISS Indexing
    vstore = VectorStoreManager()
    vstore.add_chunks(chunks)
    print(f"Indexed {vstore.total_chunks} chunks into FAISS. Dimension: {vstore.dimension}")
    
    rag = FinVerifyRAG(vector_store=vstore)
    llm = LLMGenerator(provider="gemini")
    
    # 3. Test Answerable Question
    answerable_q = "What was the total net revenue in Q3 2024, and what was the net income?"
    print(f"\n--- Testing Answerable Question: '{answerable_q}' ---")
    res1 = rag.query(answerable_q, llm_generator=llm, top_k=5)
    print(f"Confidence Level: {res1['confidence_level']}")
    print(f"Confidence Score: {res1['confidence_score']}")
    print(f"Top Similarity: {res1['top_similarity']}")
    print(f"Is Sufficient: {res1['is_sufficient']}")
    print(f"Citations Count: {len(res1['citations'])}")
    for cite in res1['citations']:
        print(f"  * [Doc: {cite['document_name']}, Page: {cite['page_number']}, Chunk: {cite['chunk_id']}, Score: {cite['similarity_score']}]")
    print(f"Answer: {res1['answer'][:150]}...")
    
    assert res1['is_sufficient'] is True, "Answerable question should be deemed sufficient"
    assert res1['confidence_level'] in ["High", "Medium"], "Confidence should be High or Medium"
    assert len(res1['citations']) > 0, "Citations must be present"
    assert "document_name" in res1['citations'][0]
    assert "page_number" in res1['citations'][0]
    assert "chunk_id" in res1['citations'][0]
    print(">>> Answerable Question Test PASSED! <<<")
    
    # 4. Test Unanswerable Question
    unanswerable_q = "What were the total commercial jet aircraft deliveries to Scandinavian Airlines in 1994?"
    print(f"\n--- Testing Unanswerable Question: '{unanswerable_q}' ---")
    res2 = rag.query(unanswerable_q, llm_generator=llm, top_k=5)
    print(f"Confidence Level: {res2['confidence_level']}")
    print(f"Confidence Score: {res2['confidence_score']}")
    print(f"Top Similarity: {res2['top_similarity']}")
    print(f"Is Sufficient: {res2['is_sufficient']}")
    print(f"Answer: {res2['answer']}")
    print(f"Exact Fallback Expected: {EXACT_FALLBACK_ANSWER}")
    
    assert res2['is_sufficient'] is False, "Unanswerable question must be marked insufficient"
    assert res2['confidence_level'] == "Insufficient Evidence", "Confidence must be Insufficient Evidence"
    assert res2['answer'] == EXACT_FALLBACK_ANSWER, f"Answer must exactly equal '{EXACT_FALLBACK_ANSWER}'"
    print(">>> Unanswerable Question Test PASSED! <<<")
    
    print("\n=== ALL PIPELINE TESTS COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_pipeline()
