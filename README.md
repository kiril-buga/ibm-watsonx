# Insurance Policy RAG on IBM watsonx

A retrieval-augmented question-answering system over multilingual insurance policy documents: Docling parses the PDFs, Milvus serves filtered vector search, and a Llama 4 model on watsonx.ai answers in the user's language with a file-and-page citation — or says it doesn't know.

https://github.com/user-attachments/assets/d3d19bf6-515c-40cb-b19c-47d07e56230a

## Challenge

Built in May 2025 for the **IBM watsonx GenAI Challenge**, on a Helvetia Insurance use case. The problem: general terms and conditions (AVB/CGA) are reissued every few years in German, French and English, so "what does my policy say about X" depends on *which* edition of *which* product you hold. Plain semantic search over the whole corpus returns the right clause from the wrong year.

The answer here is metadata-filtered retrieval — product name, edition month/year, company entity and chapter are indexed as scalar fields alongside the vectors, so a query can be scoped to one edition before the vector search runs.

Hackathon code, written under time pressure, kept public as a record of the architecture.

## Stack

- **Docling** (`HierarchicalChunker`, TableFormer, optional OCR) — PDF → structured chunks with page numbers and chapter headings preserved
- **`intfloat/multilingual-e5-large`** embeddings, with the `passage:` / `query:` prefixes the model expects
- **Milvus** on watsonx.data — IVF_FLAT + cosine (`nlist` 1024, `nprobe` 10), plus INVERTED indexes on the scalar metadata fields for pre-filtering
- **watsonx.ai** running `meta-llama/llama-4-maverick-17b-128e-instruct-fp8`, greedy decoding at temperature 0
- **LangChain** retrieval chain, **Streamlit** front end
- **Flask** microservice on IBM Code Engine, wrapped as OpenAPI extensions so watsonx Prompt Lab can call it as a tool

## Layout

| Path | What it is |
|---|---|
| `UI/` | Streamlit app — product/edition dropdowns, chat, sources panel |
| `Milvus_Search_App/` | Flask service: `/search`, `/compare`, `/get_product_names`, `/ping` |
| `milvusExtension/`, `prompt_template_extension/` | OpenAPI specs exposing the above to watsonx |
| `Prototyping/` | Notebooks for ingestion, chunking, indexing and metadata backfill |

## Running it

Needs an IBM Cloud account with watsonx.ai and a Milvus instance on watsonx.data — there is no local fallback.

```bash
cd UI
pip install -r requirements.txt
streamlit run app.py
```

with a `.env` alongside it containing:

```
API_KEY=            # IBM Cloud IAM API key
PROJECT_ID=         # watsonx.ai project
MILVUS_HOST=        # watsonx.data Milvus host
MILVUS_PORT=
MILVUS_USER=ibmlhapikey
MILVUS_PASSWORD=    # IBM Cloud API key
```

The retrieval service runs separately:

```bash
cd Milvus_Search_App
pip install -r requirements.txt
python3 app.py          # listens on :8080; GET /ping first to warm the embedding model
./example_search.sh     # sample request
```

First request is slow — `multilingual-e5-large` loads on demand.

## What it does

The Streamlit app builds a Milvus boolean expression from the dropdown selections (`product_name`, `product_month`/`product_year`, `company_entity`), passes it as `expr` to the retriever, and prepends each chunk's metadata header to its text before the chunks reach the model. The system prompt requires answers to come only from the supplied context, cites as *(file, p. page)*, mirrors the question's language, and falls back to a fixed refusal string when the context doesn't contain the answer.

`/compare` runs the same retrieval twice against two different editions of a product and hands both sets of passages to a comparison prompt template — the "what changed between the 2014 and 2023 terms" case.

## Corpus

Two document sets were ingested during the challenge, both stored as pickled chunk lists in `Prototyping/`:

| | Documents | Chunks |
|---|---|---|
| Helvetia | 184 PDFs | 21,145 |
| Phoenix | 94 PDFs | 5,376 |
