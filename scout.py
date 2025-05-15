"""
Usage:
  python scout.py --query "LLM energy use in data centers" --k 12
Generates out/2025-05-14_llm_energy_brief.md
"""

import argparse, os, json, pathlib, datetime as dt
from typing import List
import openai, requests, numpy as np
from utils import embed, chunk, build_index, search, time_decay

ARXIV_API = "https://export.arxiv.org/api/query?search_query={}&sortBy=submittedDate&sortOrder=descending&max_results={}"

def fetch_papers(query: str, n: int = 15) -> List[dict]:
    question = requests.utils.quote(query)
    
    #Requesting response from arXiv
    xml = requests.get(ARXIV_API.format(question, n), timeout=20).text

    # arXiv API returns an Atom feed and they are split into entries
    entries = xml.split("<entry>")[1:]
    papers = []
    
    # Extract fields from each paper entry
    for e in entries:
        #Function to get info in between tags
        get = lambda tag: (e.split(f"<{tag}>")[1].split(f"</{tag}>")[0]).strip()
        papers.append({
            "id": get("id"),
            "title": get("title").replace("\n", " "),
            "summary": get("summary").replace("\n", " "),
            "published": get("published")[:10] #Truncates to show only YYYY-MM-DD
        })
    return papers

def build_vector_store(chunks: List[str]) -> tuple:
    vecs = embed(chunks)
    index = build_index(vecs)
    return index, vecs

def retrieve_evidence(papers: List[dict], user_q: str, top_k: int = 6):
    all_chunks, meta = [], []
    # Breaking down papers into smaller chunks
    for p in papers:
        for c in chunk(p["title"] + ". " + p["summary"]):
            all_chunks.append(c)
            meta.append(p)
    
    #Build idex and ember user query
    index, _ = build_vector_store(all_chunks)
    q_emb = embed([user_q])

    '''
    Finds top_k most similar chunks and returns their index position 
    and similarity score
    '''
    idxs, sims = search(index, q_emb, top_k)
    evidence = []

    #For chunks, store the text, sim score, timeliness, and paper they're in
    for i, s in zip(idxs, sims):
        p = meta[i]
        evidence.append({
            "text": all_chunks[i],
            "similarity": float(s),
            "decay": time_decay(p["published"]),
            "paper": p
        })
    # re-rank by sim × decay, return k most useful (I choose 6)
    evidence.sort(key=lambda x: x["similarity"] * x["decay"], reverse=True)
    return evidence[:top_k]

PROMPT = """
You are an academic assistant.
Given a user query and a set of evidence snippets (title + abstract fragments),
produce a concise literature brief in Markdown:

1. One-sentence overall takeaway.
2. Bullet list of key findings (≤7 bullets).  Each bullet ends with an in-line citation [1].
3. Paragraph “Gaps & open questions”.
4. At bottom, full bibliography list numbered [1]–[N] with title, authors if available, arXiv link, and publication date.

Strictly ground every claim to the provided snippets; do not hallucinate papers.
"""

def generate_brief(query: str, evidence: List[dict]) -> str:
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Query: {query}\n\nEvidence:\n" +
         "\n\n".join([f"<doc>{e['text']}</doc>" for e in evidence])}
    ]
    resp = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return resp.choices[0].message.content

# Just saving the generated brief into a markdown file + returning file path
# I had to look up how to do this, but open to feedback on better methods!
def save_md(content: str, query: str) -> str:
    fn = f"out/{dt.date.today()}_{query.lower().replace(' ', '_')}.md"
    pathlib.Path("out").mkdir(exist_ok=True)
    with open(fn, "w") as f:
        f.write(content)
    return fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Research question")
    parser.add_argument("--k", type=int, default=12, help="# of arXiv papers to fetch")
    args = parser.parse_args()

    papers = fetch_papers(args.query, args.k)
    evidence = retrieve_evidence(papers, args.query, top_k=6)
    brief = generate_brief(args.query, evidence)
    path = save_md(brief, args.query)
    print(f"Brief saved to {path}")