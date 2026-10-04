"""
FinVerify AI - Document Ingestion & Chunking Module
Extracts text from uploaded PDF files, preserves document name, page number,
and chunk ID, and chunks documents using recursive character splitting.
"""

from __future__ import annotations
import io
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
from pypdf import PdfReader


@dataclass
class DocumentChunk:
    """Represents a single text chunk with provenance metadata."""
    chunk_id: str
    document_name: str
    page_number: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "content": self.content,
            "metadata": self.metadata,
        }


class RecursiveTextChunker:
    """
    Splits text into chunks of target size and overlap, respecting natural
    boundaries: paragraphs (\n\n), line breaks (\n), sentences (. ), and words ( ).
    """
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        if not separators:
            # Fallback to hard character slice if no separators left
            return self._slice_with_overlap(text)

        sep = separators[0]
        remaining_separators = separators[1:]

        if sep == "":
            splits = list(text)
        else:
            # Keep separator or split by it
            splits = text.split(sep)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            good_splits.append(s)

        current_chunk: List[str] = []
        current_len = 0

        for segment in good_splits:
            segment_len = len(segment) + (len(sep) if sep else 0)
            if current_len + segment_len <= self.chunk_size:
                current_chunk.append(segment)
                current_len += segment_len
            else:
                if current_chunk:
                    joined = sep.join(current_chunk).strip()
                    if len(joined) > self.chunk_size and remaining_separators:
                        final_chunks.extend(self._recursive_split(joined, remaining_separators))
                    elif joined:
                        final_chunks.append(joined)

                    # Build overlap from recent elements
                    overlap_chunk: List[str] = []
                    overlap_len = 0
                    for prev_seg in reversed(current_chunk):
                        if overlap_len + len(prev_seg) + len(sep) <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev_seg)
                            overlap_len += len(prev_seg) + len(sep)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_len = overlap_len

                if segment_len > self.chunk_size:
                    if remaining_separators:
                        sub_chunks = self._recursive_split(segment, remaining_separators)
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.extend(self._slice_with_overlap(segment))
                else:
                    current_chunk.append(segment)
                    current_len += segment_len

        if current_chunk:
            joined = sep.join(current_chunk).strip()
            if joined:
                if len(joined) > self.chunk_size and remaining_separators:
                    final_chunks.extend(self._recursive_split(joined, remaining_separators))
                else:
                    final_chunks.append(joined)

        return [c.strip() for c in final_chunks if c.strip()]

    def _slice_with_overlap(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step
        return chunks


def clean_pdf_text(text: str) -> str:
    """
    Cleans up raw extracted PDF text.
    Solves the vertical text issue where PDFs export single words per line.
    Unwraps fragmented lines into continuous horizontal paragraphs.
    """
    if not text:
        return ""
    # Remove null bytes and non-printable control characters
    text = text.replace("\x00", "")
    # Normalize multiple tabs/spaces
    text = re.sub(r"[ \t]+", " ", text)
    
    # Split into lines
    raw_lines = [l.strip() for l in text.split("\n")]
    non_empty = [l for l in raw_lines if l]
    
    if not non_empty:
        return ""
        
    # Check if lines are unnaturally short (e.g. single-word line breaks as in slides/cheatsheets)
    avg_words_per_line = sum(len(l.split()) for l in non_empty) / len(non_empty)
    
    # If document has many short lines (average words < 4), unwrap into horizontal text
    if avg_words_per_line < 4.0:
        paragraphs = []
        current_words = []
        for line in raw_lines:
            if not line:
                if current_words:
                    paragraphs.append(" ".join(current_words))
                    current_words = []
                continue
            
            # If line is a list bullet or numbered item
            if re.match(r"^([•\-\*]|\d+[\.\)])\s*", line):
                if current_words:
                    paragraphs.append(" ".join(current_words))
                    current_words = []
            current_words.append(line)
            
        if current_words:
            paragraphs.append(" ".join(current_words))
        
        text = "\n\n".join(paragraphs)
    else:
        # Standard unwrapping: join lines that don't end with sentence-ending punctuation
        paragraphs = []
        current_line = []
        for line in raw_lines:
            if not line:
                if current_line:
                    paragraphs.append(" ".join(current_line))
                    current_line = []
                continue
            if current_line:
                prev = current_line[-1]
                if prev.endswith((".", "!", "?", ":")) or re.match(r"^[•\-\*]", prev):
                    paragraphs.append(" ".join(current_line))
                    current_line = [line]
                else:
                    current_line.append(line)
            else:
                current_line.append(line)
        if current_line:
            paragraphs.append(" ".join(current_line))
        text = "\n\n".join(paragraphs)

    # Normalize multiple whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_chunks_from_pdf(
    file_bytes_or_path: Union[str, bytes, io.BytesIO],
    document_name: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[DocumentChunk]:
    """
    Extracts pages from a PDF and chunks each page while strictly preserving:
    - document_name
    - page_number (1-indexed)
    - chunk_id (e.g. "{doc_slug}_p{page_num}_c{chunk_idx}")
    """
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        pdf_stream = io.BytesIO(file_bytes_or_path)
    elif isinstance(file_bytes_or_path, io.BytesIO):
        pdf_stream = file_bytes_or_path
    else:
        pdf_stream = open(file_bytes_or_path, "rb")

    reader = PdfReader(pdf_stream)
    chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Safe document slug for IDs
    clean_doc_slug = re.sub(r"[^\w\-.]", "_", document_name.lower().replace(".pdf", ""))
    chunks: List[DocumentChunk] = []

    global_chunk_idx = 0
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""
        cleaned_text = clean_pdf_text(raw_text)

        if not cleaned_text:
            continue

        page_chunks = chunker.split_text(cleaned_text)
        for p_c_idx, c_text in enumerate(page_chunks):
            chunk_id = f"{clean_doc_slug}_p{page_num}_c{p_c_idx+1}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=document_name,
                    page_number=page_num,
                    content=c_text,
                    metadata={
                        "char_count": len(c_text),
                        "global_chunk_index": global_chunk_idx,
                        "page_chunk_index": p_c_idx + 1,
                    },
                )
            )
            global_chunk_idx += 1

    return chunks


def ingest_multiple_pdfs(
    files: List[tuple[str, Union[bytes, io.BytesIO, str]]],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[DocumentChunk]:
    """
    Ingests multiple PDF files and returns an aggregated list of DocumentChunks.
    `files` is a list of tuples: (document_name, file_content)
    """
    all_chunks: List[DocumentChunk] = []
    for doc_name, content in files:
        doc_chunks = extract_chunks_from_pdf(
            content,
            document_name=doc_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks
