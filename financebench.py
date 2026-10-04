"""
FinVerify AI - PatronusAI FinanceBench Benchmark Integration Module
Loads and indexes the PatronusAI/financebench dataset (150 real SEC filing questions
and evidence across 32 premier companies: Apple, Microsoft, Amazon, 3M, Coca-Cola, etc.)
"""

from __future__ import annotations
import os
import json
import re
from typing import List, Dict, Any, Optional
from ingest import DocumentChunk, clean_pdf_text, RecursiveTextChunker

DATASET_FILE = os.path.join(os.path.dirname(__file__), "financebench_data.jsonl")
HF_DATASET_URL = "https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/financebench_merged.jsonl"


def ensure_dataset_downloaded() -> str:
    """Ensures financebench_data.jsonl is present locally."""
    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) == 0:
        import urllib.request
        req = urllib.request.Request(HF_DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(DATASET_FILE, "wb") as f:
            f.write(resp.read())
    return DATASET_FILE


class FinanceBenchDataset:
    """Manages the PatronusAI/financebench benchmark dataset."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or ensure_dataset_downloaded()
        self.records: List[Dict[str, Any]] = []
        self._load_records()

    def _load_records(self) -> None:
        if not os.path.exists(self.data_path):
            return
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.records.append(json.loads(line))
                    except Exception:
                        continue

    def get_companies(self) -> List[str]:
        """Returns sorted list of unique companies in FinanceBench."""
        companies = set(r.get("company", "").strip() for r in self.records if r.get("company"))
        return sorted(list(companies))

    def get_documents(self, company: Optional[str] = None) -> List[str]:
        """Returns list of unique SEC filings for a company or all companies."""
        docs = set()
        for r in self.records:
            if company and r.get("company") != company:
                continue
            doc = r.get("doc_name")
            if doc:
                docs.add(doc)
        return sorted(list(docs))

    def get_records_by_company(self, company: str) -> List[Dict[str, Any]]:
        """Returns all benchmark QA entries for a given company."""
        return [r for r in self.records if r.get("company") == company]

    def get_record_by_id(self, fb_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific benchmark record by its financebench_id."""
        for r in self.records:
            if r.get("financebench_id") == fb_id:
                return r
        return None

    def extract_chunks(
        self,
        company: Optional[str] = None,
        chunk_size: int = 1800,
        chunk_overlap: int = 200,
    ) -> List[DocumentChunk]:
        """
        Extracts all SEC filing evidence from FinanceBench records and converts
        them into DocumentChunk objects with provenance metadata and horizontal text.
        Preserves intact financial statements without table fragmentation.
        """
        chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks: List[DocumentChunk] = []
        seen_evidence_hashes = set()

        records_to_process = self.get_records_by_company(company) if company else self.records

        global_idx = 0
        for rec in records_to_process:
            comp_name = rec.get("company", "Unknown")
            doc_name = rec.get("doc_name", f"{comp_name}_SEC_Filing")
            if not doc_name.endswith(".pdf"):
                doc_file_name = f"{doc_name}.pdf"
            else:
                doc_file_name = doc_name

            fb_id = rec.get("financebench_id", "")
            gt_answer = rec.get("answer", "")
            justification = rec.get("justification", "")
            q_type = rec.get("question_type", "")
            evidence_list = rec.get("evidence", [])

            for ev in evidence_list:
                ev_text = ev.get("evidence_text", "")
                page_num = ev.get("evidence_page_num", 1)
                if not ev_text:
                    continue

                # Un-fragment and clean PDF table/text
                cleaned = clean_pdf_text(ev_text)
                if not cleaned:
                    continue

                # Deduplicate evidence blocks across questions
                ev_hash = hash((doc_name, page_num, cleaned[:120]))
                if ev_hash in seen_evidence_hashes:
                    continue
                seen_evidence_hashes.add(ev_hash)

                doc_slug = re.sub(r"[^\w\-.]", "_", doc_name.lower().replace(".pdf", ""))
                
                # If evidence block is under chunk_size, keep as single intact statement
                if len(cleaned) <= chunk_size:
                    sub_chunks = [cleaned]
                else:
                    sub_chunks = chunker.split_text(cleaned)

                for c_idx, c_text in enumerate(sub_chunks):
                    # Clean continuous horizontal text (no vertical wrap)
                    c_clean = " ".join(c_text.split())
                    chunk_id = f"{doc_slug}_p{page_num}_c{c_idx+1}"
                    header = f"[{comp_name} | SEC Filing: {doc_name} | Page {page_num}] "
                    
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_name=doc_file_name,
                            page_number=page_num,
                            content=header + c_clean,
                            metadata={
                                "company": comp_name,
                                "financebench_id": fb_id,
                                "ground_truth_answer": gt_answer,
                                "justification": justification,
                                "question_type": q_type,
                                "doc_type": rec.get("doc_type", "SEC 10-K"),
                                "doc_period": rec.get("doc_period", ""),
                            },
                        )
                    )
                    global_idx += 1

        return chunks
