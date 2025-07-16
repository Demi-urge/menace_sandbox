# TextResearchBot

`TextResearchBot` scrapes webpages and PDF files before producing short summaries. When `gensim.summarization` is available the bot uses it directly. If gensim is missing or fails, the fallback now tokenizes sentences via `nltk` when possible and ranks them with TF‑IDF weights from `sklearn`. The highest scoring sentences are returned according to the supplied ratio, yielding concise summaries even without external libraries.
