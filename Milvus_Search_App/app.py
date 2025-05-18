from flask import Flask, request, jsonify
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
import os
import logging
from datetime import datetime
from difflib import unified_diff

logging.basicConfig(level=logging.INFO)

password = os.environ.get("MILVUS_PASSWORD")
user = os.environ.get("MILVUS_USER", "ibmlhapikey")

app = Flask(__name__)

model_name = "intfloat/multilingual-e5-large"
print("🔄 Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("✅ Model loaded.")

@app.route("/ping", methods=["GET"])
def ping():
    embed("warmup")  # silently loads everything into memory
    return "Warmed up", 200

def embed(text):
    input_text = f"query: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0]
    return embeddings[0].tolist()

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.json
        query = data["query"]
        collection_name = data["collection_name"]
        output_fields = data["output_fields"]
        filter_expr = data["filter"]
        top_k = data["top_k"]

        load_dotenv()

        connections.connect(
            host="102092af-5474-4a42-8dc2-35bb05ffdd0e.cvgfjtof0l91rq0joaj0.lakehouse.appdomain.cloud",
            port="31574",
            user=os.environ.get("MILVUS_USER"),
            password=os.environ.get("MILVUS_PASSWORD"),
            secure=True
        )

        emb_query = embed(query)
        collection = Collection(collection_name)
        #collection.load()
        #print(collection.num_entities, flush=True)

        results = collection.search(
            data=[emb_query],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )
        #print(results)

        output = [{"id": hit.id, "score": hit.distance, "text": hit.entity.get("text"), "file_name": hit.entity.get("file_name"), "product_year": hit.entity.get("product_year"), "product_name": hit.entity.get("product_name")} for hit in results[0]]

        return jsonify({"results": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare", methods=["POST"])
def compare_definitions():
    try:
        data = request.json
        query = data["query"]
        collection_name = data["collection_name"]
        output_fields = data.get("output_fields")
        vector_field = data.get("vector_field", "vector")
        product_name = data["product_name"]
        topic_field = data.get("topic_field", None) #--> "chapter" for example
        topic_value = data.get("topic_value", None) # --> "a specific chapter name"
        years = data["years"]  # e.g. [2014, 2023] --> possibility of comparing more than two years...
        top_k = data.get("top_k", 3)

        load_dotenv()

        connections.connect(
            host="102092af-5474-4a42-8dc2-35bb05ffdd0e.cvgfjtof0l91rq0joaj0.lakehouse.appdomain.cloud",
            port="31574",
            user=os.environ.get("MILVUS_USER"),
            password=os.environ.get("MILVUS_PASSWORD"),
            secure=True
        )

        response = compare_definitions_by_year(query, collection_name, vector_field, output_fields, product_name,
                                               topic_field, topic_value, years, top_k)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# used to compare two policies with the same product name but different dates (e.g. 2014 vs 2023)
# it searches for the most recent policy before the given date and compares it to the query
@app.route("/compare_by_date", methods=["POST"])
def compare_by_date():
    try:
        data = request.json
        query = data["query"]
        collection_name = data["collection_name"]
        product_name = data["product_name"]
        dates = data["dates"]
        output_fields = data.get("output_fields", ["text", "file_name", "product_year", "product_month"])
        vector_field = data.get("vector_field", "vector")
        top_k = data.get("top_k", 3)

        load_dotenv()

        connections.connect(
            host="102092af-5474-4a42-8dc2-35bb05ffdd0e.cvgfjtof0l91rq0joaj0.lakehouse.appdomain.cloud",
            port="31574",
            user=os.environ.get("MILVUS_USER"),
            password=os.environ.get("MILVUS_PASSWORD"),
            secure=True
        )

        collection = Collection(collection_name)
        query_vector = embed(query)
        results = {}

        for date_str in dates:
            version_info = find_most_recent_policy_before(date_str, collection, product_name)
            if not version_info:
                results[f"context_{date_str}"] = "No matching version found"
                results[f"title_{date_str}"] = "N/A"
                continue

            year = version_info["product_year"]
            month = version_info["product_month"]

            filter_expr = f'product_year == {year} && product_month == {month} && product_name == "{product_name}"'

            search_result = collection.search(
                data=[query_vector],
                anns_field=vector_field,
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )

            chunks = []
            file_title = None
            for hit in search_result[0]:
                txt = hit.entity.get("text", "").strip()
                if txt:
                    chunks.append(txt)
                if not file_title:
                    file_title = hit.entity.get("file_name", f"{year}_{month:02d}")

            label = f"{year}_{month:02d}"
            results[f"context_{label}"] = "\n---\n".join(chunks)
            results[f"title_{label}"] = file_title or "Unknown"

        return jsonify({"comparison": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug/latest_policy", methods=["GET"])
def debug_latest_policy():
    date_str = request.args.get("date", "04.2015")
    product_name = request.args.get("product_name", "Absicherungsplan")
    connections.connect(
        host="102092af-5474-4a42-8dc2-35bb05ffdd0e.cvgfjtof0l91rq0joaj0.lakehouse.appdomain.cloud",
        port="31574",
        user=os.environ.get("MILVUS_USER"),
        password=os.environ.get("MILVUS_PASSWORD"),
        secure=True
    )

    collection_name = Collection("docling_helvetia")
    collection_name.load()

    result = find_most_recent_policy_before(date_str, collection_name, product_name)
    result["original_date"] = date_str
    return jsonify(result or {"error": "No match"})

def find_most_recent_policy_before(user_date_str, collection, product_name):
    user_date = datetime.strptime(user_date_str, "%m.%Y")

    # Get all versions for this product
    docs = collection.query(
        expr=f'product_name == "{product_name}"',
        output_fields=["product_year", "product_month"]
    )

    # Extract all policy dates
    valid_versions = []
    for d in docs:
        try:
            y = int(d["product_year"])
            m = int(d["product_month"])
            doc_date = datetime(y, m, 1)
            if doc_date <= user_date:
                valid_versions.append((doc_date, y, m))
        except:
            continue  # skip bad records

    if not valid_versions:
        return None  # no applicable policy found

    # Sort and return the latest one before user date
    valid_versions.sort(reverse=True)
    _, best_year, best_month = valid_versions[0]
    return {"product_year": best_year, "product_month": best_month}



def compare_definitions_by_year(query, collection_name, vector_field, output_fields, product_name, topic_field,
                                topic_value, years, top_k=10):
    """
    Perform two filtered searches (one per year) in Milvus and format the results for comparison.

    Args:
        query: User query string
        collection_name: Name of Milvus collection
        vector_field: Name of the vector field in Milvus
        output_fields: List of non-vector fields to return --> list
        product_name: Filter product name (e.g. 'Absicherungsplan') --> string
        topic_field: Optional metadata field like 'chapter' --> string
        topic_value: Optional topic or section to narrow scope
        years: List of two years [year1, year2]
        top_k: How many top results to return per year
    """
    results = {}
    collection = Collection(collection_name)
    query_vector = embed(query)
    for year in years:
        filter_parts = [f'product_year == {year}', f'product_name == "{product_name}"']
        if topic_field and topic_value:
            filter_parts.append(f'{topic_field} == "{topic_value}"')
        filter_expr = " && ".join(filter_parts)

        search_result = collection.search(
            data=[query_vector],
            anns_field=vector_field,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )

        # Format and concatenate chunks
        text_chunks = []
        metadata_fields = {}
        for hit in search_result[0]:
            chunk_text = hit.entity.get("text", "").strip()
            if chunk_text:
                text_chunks.append(chunk_text)
            # capture metadata once
            if not metadata_fields:
                for key in ["product_name", "product_year", "file_name"]:
                    value = hit.entity.get(key)
                    if value is not None:
                        metadata_fields[key] = value

        results[f"context_{year}"] = "\n---\n".join(text_chunks)
        results[f"metadata_{year}"] = metadata_fields

    return {
            "context": results
    }

@app.route("/summarize_policy", methods=["POST"])
def summarize_policy():
    try:
        data = request.json
        query = data["query"]
        collection_name = data["collection_name"]
        output_fields = data["output_fields"]
        filter_expr = data["filter"]
        top_k = data["top_k"]

        load_dotenv()

        connections.connect(
            host="102092af-5474-4a42-8dc2-35bb05ffdd0e.cvgfjtof0l91rq0joaj0.lakehouse.appdomain.cloud",
            port="31574",
            user=os.environ.get("MILVUS_USER"),
            password=os.environ.get("MILVUS_PASSWORD"),
            secure=True
        )

        emb_query = embed(query)
        collection = Collection(collection_name)

        results = collection.search(
            data=[emb_query],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )

        # Extract the texts from the hits
        texts = []
        for hit in results[0]:
            text = hit.entity.get("text")
            if text:
                texts.append(text)

        # Generate summary
        summary = summarize_with_embeddings(query, texts, top_k=3)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def summarize_with_embeddings(query, texts, top_k=3):
    """
    Summarize texts by embedding and picking top_k most relevant chunks to the query.
    """
    query_emb = embed(query)

    # Embed each text chunk
    text_embs = [embed(t) for t in texts]

    # Compute cosine similarity between query_emb and each text_emb
    def cosine_sim(a, b):
        a = torch.tensor(a)
        b = torch.tensor(b)
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    similarities = [(cosine_sim(query_emb, emb), idx) for idx, emb in enumerate(text_embs)]
    # Sort descending by similarity
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Pick top_k chunks
    top_chunks = [texts[idx] for _, idx in similarities[:top_k]]

    # Join chunks as summary
    summary = "\n\n".join(top_chunks)
    return summary

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

