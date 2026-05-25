"""Deterministic top-3 article retrieval using TF-IDF cosine similarity.

The corpus is built from articles (title + category + content).
Queries are built from preprocessed ticket text (subject + message).
Ties are broken by article_id lexicographic order to ensure full
reproducibility across runs.
"""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import CandidateArticle, PreprocessedTicket, RawArticle, RetrievalResult


def _article_text(article: RawArticle) -> str:
    return f"{article.title} {article.category} {article.content}"


def _ticket_text(ticket: PreprocessedTicket) -> str:
    return f"{ticket.subject_for_processing} {ticket.message_for_processing}"


def retrieve_candidates(
    preprocessed_tickets: List[PreprocessedTicket],
    articles: List[RawArticle],
    top_k: int = 3,
) -> List[RetrievalResult]:
    """Return top_k candidate articles per ticket, scored by TF-IDF cosine similarity.

    Sorting is deterministic: descending score, then ascending article_id on ties.
    """
    if not articles:
        raise ValueError("Article corpus is empty – cannot perform retrieval.")
    if not preprocessed_tickets:
        return []

    # Sort articles by ID to guarantee a fixed column order in the matrix
    sorted_articles = sorted(articles, key=lambda a: a.article_id)
    article_texts = [_article_text(a) for a in sorted_articles]
    ticket_texts = [_ticket_text(t) for t in preprocessed_tickets]

    # Fit on the union of articles and tickets so ticket tokens that
    # appear in articles are scored correctly
    vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,      # log(1+tf) dampening for long docs
        min_df=1,
        ngram_range=(1, 2),     # uni- and bigrams
    )
    all_texts = article_texts + ticket_texts
    vectorizer.fit(all_texts)

    article_matrix = vectorizer.transform(article_texts)
    ticket_matrix = vectorizer.transform(ticket_texts)

    # cosine_similarity returns shape (n_tickets, n_articles)
    sim_matrix: np.ndarray = cosine_similarity(ticket_matrix, article_matrix)

    results: List[RetrievalResult] = []
    for i, ticket in enumerate(preprocessed_tickets):
        scores = sim_matrix[i]  # 1-D array over sorted_articles
        # Build list of (article_id, score) and apply deterministic sort
        scored = [
            (sorted_articles[j].article_id, float(scores[j]))
            for j in range(len(sorted_articles))
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))  # score desc, id asc
        top = scored[:top_k]

        results.append(
            RetrievalResult(
                ticket_id=ticket.ticket_id,
                candidate_articles=[
                    CandidateArticle(
                        article_id=aid,
                        score=round(score, 6),
                    )
                    for aid, score in top
                ],
            )
        )

    return results
