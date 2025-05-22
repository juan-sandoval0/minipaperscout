# Mini-PaperScout

I created a research assistant that summarizes recent academic papers in response to a user-defined question. It pulls papers from arXiv, ranks them based on semantic and temporal relevance, and generates a brief from the text excerpts with citations. 

I built it because I was working on a research project and noticed that when I asked ChatGPT for help, it often gave me vague answers with hallucinated citations or sources that didn’t exist. I just wanted a way to quickly get up to speed on a topic, with real links and summaries I could trust. So this was my attempt to build a better tool for that.

The project uses a basic RAG pipeline and runs entirely locally.

---

## How It Works

1. You type in a research question (e.g. "How are large language models being optimized for energy efficiency?")
2. The script fetches the most recent papers from arXiv related to that question.
3. It splits and embeds the abstracts using OpenAI's embedding API.
4. The most relevant excerpts are retrieved, ranked with a time-decay score (to favor newer work), and passed into GPT-4o for summarization.
5. The output is a Markdown file with:
   - A one-line summary
   - A bullet list of key findings
   - A paragraph on open questions or research gaps
   - A bibliography with links and publication dates
