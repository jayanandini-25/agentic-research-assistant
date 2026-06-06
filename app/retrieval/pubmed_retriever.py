"""
app/retrieval/pubmed_retriever.py  (fixed)
 
Key fixes
---------
1. Uses preprocessor for optimised medical query.
2. Relevance post-filter against original question.
3. Removed hard-coded MEDICAL_KEYWORDS gate — the preprocessor already
   extracts the right terms; we rely on PubMed returning 0 results naturally
   for non-medical topics rather than pre-blocking.
   (Keeping a light keyword gate to avoid wasting quota.)
"""
 
import asyncio
import httpx
from typing import List, Dict
 
from core.logger import setup_logger
from app.retrieval.query_preprocessor import build_source_queries, relevance_score
 
logger = setup_logger(__name__)
 
PUBMED_SEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
 
MEDICAL_SIGNALS = {
    "disease", "drug", "medicine", "health", "cancer", "virus", "bacteria",
    "treatment", "clinical", "patient", "medical", "biology", "gene", "protein",
    "vaccine", "therapy", "diagnosis", "symptom", "hospital", "covid", "infection",
    "disorder", "syndrome", "pharmaceutical", "neurological", "cardiac",
}
 
 
def _is_medical(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in MEDICAL_SIGNALS)
 
 
class PubMedRetriever:
 
    def __init__(self):
        logger.info("PubMedRetriever initialized")
 
    async def retrieve(self, question: str, max_results: int = 5) -> List[Dict]:
        if not _is_medical(question):
            logger.info(f"PubMed skipped (non-medical): '{question[:60]}'")
            return []
 
        search_phrase = build_source_queries(question)["pubmed"]
        logger.info(f"PubMed | phrase='{search_phrase}' (from: '{question[:60]}')")
 
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                # Step 1: get IDs
                search_resp = await client.get(
                    PUBMED_SEARCH_URL,
                    params={
                        "db":      "pubmed",
                        "term":    search_phrase,
                        "retmax":  max_results * 2,
                        "retmode": "json",
                        "sort":    "relevance",
                    },
                )
                search_resp.raise_for_status()
                ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    logger.debug("PubMed: no results")
                    return []
 
                # Step 2: get summaries
                summary_resp = await client.get(
                    PUBMED_SUMMARY_URL,
                    params={
                        "db":     "pubmed",
                        "id":     ",".join(ids),
                        "retmode":"json",
                    },
                )
                summary_resp.raise_for_status()
                articles = summary_resp.json().get("result", {})
 
            results: List[Dict] = []
            for uid in ids:
                if len(results) >= max_results:
                    break
                article = articles.get(uid, {})
                title   = article.get("title", "")
                authors = ", ".join(
                    a.get("name", "") for a in article.get("authors", [])[:3]
                )
                pub_date = article.get("pubdate", "")
                journal  = article.get("source", "")
 
                content = (
                    f"{title}\n\nAuthors: {authors}\n"
                    f"Journal: {journal}\nPublished: {pub_date}\n"
                    f"PubMed ID: {uid}"
                )
 
                score = relevance_score(question, title, content)
                if score < 0.15:
                    logger.debug(f"PubMed dropped (score={score:.2f}): '{title[:60]}'")
                    continue
 
                results.append({
                    "source":  "pubmed",
                    "title":   title,
                    "url":     f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                    "content": content,
                    "score":   score,
                })
 
            logger.info(f"PubMed returned {len(results)} results")
            return results
 
        except Exception as e:
            logger.error(f"PubMed failed: {e}")
            return []