import os

# import requests, re
from typing import Any, Dict, List

# from sentence_transformers.util import cos_sim
import wikipedia as w

from llm.embeddings import embed_query, embed_texts

w.set_lang("en")

import numpy as np
import wikipediaapi
from Bio import Entrez
from tools.tool_utils import chunk_text

ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
# MAX_CHUNK_TOKENS = os.getenv('MAX_CHUNK_TOKENS')
# CHUNK_OVERLAP = os.getenv('CHUNK_OVERLAP')

"""
Need an end chat tool maybe?

#@make_async_background
https://gofastmcp.com/servers/context
https://gofastmcp.com/servers/progress
https://medium.com/@adnanmasood/optimizing-chunking-embedding-and-vectorization-for-retrieval-augmented-generation-ea3b083b68f7
https://people.duke.edu/~ccc14/pcfb/biopython/BiopythonEntrez.html




PubMed efetch:

retmode=xml, rettype=abstract → PubMed XML for abstracts.

retmode=text, rettype=medline → classic MEDLINE/plain-text.

retmode=xml, rettype=uilist → IDs in XML.

"""


# WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_API = wikipediaapi.Wikipedia(
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,  # plain text (no HTML)
    user_agent="MediRAG-LLM-Server/0.1 (https://github.com/dvrk-dvys/wiki_rag)",
)


ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")


class WikipediaTool:
    """Live Wikipedia search with OpenAI embedding similarity."""

    def __init__(self):
        pass

    def search_wiki_titles(self, query, limit):
        """Search to get candidate article titles."""
        return w.search(query, results=limit)[:limit]

    def get_wiki_url(self, title):
        """Fetch URL of a page."""
        try:
            page = WIKI_API.page(title)
            print(f"Fetching page: {title}")
        except Exception as e:
            print(f"Error fetching page: {e}")
            return ""
        return page.fullurl

    def get_plaintext_wiki(self, title):
        """Fetch plain text of a page (lead + body) as one big string."""
        try:
            page = WIKI_API.page(title)
            print(f"Fetching page: {title}")
        except Exception as e:
            print(f"Error fetching page: {e}")
            return ""
        return page.text.strip() if page.exists() else ""

    def wiki_semantic_search(self, query, top_k=5):
        """
        Retrieve short, relevant Wikipedia passages for background/definitions.
        Returns: [{id,title,text,rrf_score,source_type}]
        """
        titles = self.search_wiki_titles(query, limit=5)
        if not titles:
            return []

        q_vec = np.array(embed_query(query), dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

        candidates = []
        for title in titles:
            fulltext = self.get_plaintext_wiki(title)
            url = self.get_wiki_url(title)
            chunks = chunk_text(fulltext)

            # score each chunk by cosine similarity
            #! note: L2-normalized, then the dot product = cosine similarity.
            #! If they are not normalized, the dot product will overweight long vectors (magnitude), whereas cosine similarity divides by the norms to measure the angle only.
            ch_vecs = np.array(embed_texts(chunks), dtype=np.float32)
            norms = np.linalg.norm(ch_vecs, axis=1, keepdims=True) + 1e-9
            ch_vecs = ch_vecs / norms
            sims = np.dot(ch_vecs, q_vec)

            for chunk, sim in zip(chunks, sims):
                candidates.append(
                    {
                        "id": f"wikipedia::{title}",
                        "title": title,
                        "url": url,
                        "text": chunk,
                        "cos_sim_score": float(sim),
                        "source_type": "wikipedia",
                    }
                )

        candidates.sort(key=lambda d: d["cos_sim_score"], reverse=True)
        return candidates[:top_k]


if __name__ == "__main__":
    wikipedia_tool = WikipediaTool()
    query = [
        "Femoral Head Necrosis?",
        "How does anesthesia block pain receptors?",
        "Femoral Head Avascular Necrosis Joint Corticosteroids",
    ]

    for q in query:
        print(f"Query: {q}")
        res = wikipedia_tool.wiki_semantic_search(q, top_k=5)
        for r in res:
            print(r)
        print("_" * 20)
