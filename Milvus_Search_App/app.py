from flask import Flask, request, jsonify
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
import os

password = os.environ.get("MILVUS_PASSWORD")
user = os.environ.get("MILVUS_USER", "ibmlhapikey")

app = Flask(__name__)

model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

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
        print(collection.num_entities, flush=True)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

