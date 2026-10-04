"""
FinVerify AI - Hallucination Guardrail & Sufficiency Engine
Enforces strict grounding, evaluates semantic and lexical evidence sufficiency,
calculates confidence tiers (High, Medium, Insufficient Evidence), and ensures
the exact fallback answer is triggered when evidence is missing.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Set, Any, Tuple

# Strict requirement: exact response string when evidence is insufficient
EXACT_FALLBACK_ANSWER = "I don't know based on the provided documents."

# Common English stopwords to ignore during lexical overlap checks
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by",
    "from", "of", "about", "into", "through", "during", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am",
    "it", "its", "as", "if", "each", "how", "why", "when", "where", "can",
    "could", "should", "would", "will", "shall", "may", "might", "must",
    "there", "here", "tell", "me", "give", "explain", "find", "show", "what's",
    "does", "much", "many", "total", "any"
}

# Negative indicator phrases from LLMs that imply lack of evidence
NEGATIVE_PATTERNS = [
    r"i do not know",
    r"i don't know",
    r"not mentioned in the provided",
    r"not mentioned in the document",
    r"not provided in the document",
    r"not stated in the document",
    r"does not contain",
    r"no mention of",
    r"no information provided",
    r"cannot be answered based on",
    r"not available in the context",
    r"insufficient information",
    r"outside the scope of the provided",
]


@dataclass
class GuardrailResult:
    """Detailed evaluation result from the guardrail check."""
    is_sufficient: bool
    confidence_level: str  # "High" | "Medium" | "Insufficient Evidence"
    confidence_score: float  # Normalized 0.0 - 1.0
    top_similarity: float
    avg_similarity: float
    keyword_overlap_ratio: float
    matched_keywords: List[str]
    query_keywords: List[str]
    explanation: str

    def get_badge_color(self) -> str:
        if self.confidence_level == "High":
            return "#10B981"  # Emerald Green
        elif self.confidence_level == "Medium":
            return "#F59E0B"  # Amber
        return "#EF4444"  # Red / Insufficient


class FinancialGuardrail:
    """
    Dual-layer validation:
    1. Vector Similarity Thresholds: Evaluates cosine similarity of top retrieved chunks.
    2. Lexical & Entity Overlap: Validates that key financial terms/entities from the question
       actually appear in the retrieved context.
    3. Post-Generation Hallucination Check: Intercepts ungrounded or hesitant LLM outputs.
    """

    def __init__(
        self,
        min_top_similarity: float = 0.38,
        min_avg_similarity: float = 0.32,
        high_confidence_similarity: float = 0.62,
    ):
        self.min_top_similarity = min_top_similarity
        self.min_avg_similarity = min_avg_similarity
        self.high_confidence_similarity = high_confidence_similarity

    def extract_keywords(self, text: str) -> List[str]:
        """Extracts meaningful terms (numbers, financial codes, alphanumeric words)."""
        tokens = re.findall(r"\b[A-Za-z0-9$%.\-]{2,}\b", text.lower())
        keywords = [t for t in tokens if t not in STOPWORDS and not t.isdigit() or len(t) > 2]
        return list(dict.fromkeys(keywords))  # preserve order, unique

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_contents: List[str],
        similarity_scores: List[float],
    ) -> GuardrailResult:
        """
        Evaluates whether the retrieved chunks contain sufficient evidence
        to ground a factual answer without hallucinating.
        """
        if not retrieved_contents or not similarity_scores:
            return GuardrailResult(
                is_sufficient=False,
                confidence_level="Insufficient Evidence",
                confidence_score=0.0,
                top_similarity=0.0,
                avg_similarity=0.0,
                keyword_overlap_ratio=0.0,
                matched_keywords=[],
                query_keywords=self.extract_keywords(query),
                explanation="No relevant document chunks found in index.",
            )

        top_sim = float(similarity_scores[0]) if similarity_scores else 0.0
        # Average of top 3 (or all if < 3)
        top_k = min(3, len(similarity_scores))
        avg_sim = float(sum(similarity_scores[:top_k]) / top_k)

        # Lexical keyword overlap
        query_kw = self.extract_keywords(query)
        combined_text = " ".join(retrieved_contents).lower()
        
        matched_kw: List[str] = []
        if query_kw:
            for kw in query_kw:
                # Direct substring match or whole word regex
                if re.search(r"\b" + re.escape(kw) + r"\b", combined_text):
                    matched_kw.append(kw)
            overlap_ratio = len(matched_kw) / len(query_kw)
        else:
            overlap_ratio = 1.0

        # Weighted composite score
        # 60% semantic similarity + 40% keyword overlap
        norm_sim = max(0.0, min(1.0, (top_sim - 0.20) / 0.65))
        composite_score = round(0.60 * norm_sim + 0.40 * overlap_ratio, 3)
        composite_score = max(0.0, min(1.0, composite_score))

        # Check sufficiency rules
        # Rule A: If top similarity is below threshold AND keyword overlap is very low -> Insufficient
        # Rule B: If top similarity is very low (< min_top_similarity - 0.05), even if random word matches -> Insufficient
        is_sufficient = True
        explanation = "Retrieved chunks contain relevant context matching the query."

        if top_sim < self.min_top_similarity and overlap_ratio < 0.35:
            is_sufficient = False
            explanation = (
                f"Top similarity ({top_sim:.2f}) and keyword overlap ({overlap_ratio:.1%}) "
                f"are below sufficiency thresholds."
            )
        elif top_sim < 0.32:
            is_sufficient = False
            explanation = f"Top similarity score ({top_sim:.2f}) is too weak to guarantee factual accuracy."
        elif len(query_kw) >= 3 and overlap_ratio < 0.20:
            is_sufficient = False
            explanation = "Core query terms are missing from the retrieved financial context."

        # Assign confidence tier
        if not is_sufficient:
            confidence_level = "Insufficient Evidence"
        elif top_sim >= self.high_confidence_similarity and overlap_ratio >= 0.50:
            confidence_level = "High"
        else:
            confidence_level = "Medium"

        return GuardrailResult(
            is_sufficient=is_sufficient,
            confidence_level=confidence_level,
            confidence_score=composite_score,
            top_similarity=round(top_sim, 3),
            avg_similarity=round(avg_sim, 3),
            keyword_overlap_ratio=round(overlap_ratio, 3),
            matched_keywords=matched_kw,
            query_keywords=query_kw,
            explanation=explanation,
        )

    def enforce_guardrail(
        self,
        llm_raw_response: str,
        guardrail_result: GuardrailResult,
    ) -> Tuple[str, GuardrailResult]:
        """
        Enforces hallucination guardrail on LLM output.
        If the evidence was deemed insufficient or the model conceded lack of evidence,
        returns the exact string: 'I don't know based on the provided documents.'
        """
        if not guardrail_result.is_sufficient:
            return EXACT_FALLBACK_ANSWER, guardrail_result

        # Check if the LLM admitted it doesn't know in the generated text
        cleaned_response = llm_raw_response.strip()
        lower_response = cleaned_response.lower()

        for pattern in NEGATIVE_PATTERNS:
            if re.search(pattern, lower_response):
                # Update guardrail result to Insufficient Evidence
                updated_result = GuardrailResult(
                    is_sufficient=False,
                    confidence_level="Insufficient Evidence",
                    confidence_score=guardrail_result.confidence_score,
                    top_similarity=guardrail_result.top_similarity,
                    avg_similarity=guardrail_result.avg_similarity,
                    keyword_overlap_ratio=guardrail_result.keyword_overlap_ratio,
                    matched_keywords=guardrail_result.matched_keywords,
                    query_keywords=guardrail_result.query_keywords,
                    explanation="LLM detected that the provided context does not contain sufficient answer facts.",
                )
                return EXACT_FALLBACK_ANSWER, updated_result

        return cleaned_response, guardrail_result
