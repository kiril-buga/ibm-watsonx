from flask import Flask, request, jsonify
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

password = os.environ.get("MILVUS_PASSWORD")
user = os.environ.get("MILVUS_USER", "ibmlhapikey")

app = Flask(__name__)

model_name = "intfloat/multilingual-e5-large"
print("🔄 Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("✅ Model loaded.")

load_dotenv()


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
    """
    This route performs a search in Milvus based on the provided query and creates a filter based on the user's input (Product name + product year)
    Takes as Input:
    {
    "query": str,
    "collection_name": "docling_helvetia",
    "product_date": "XX.XXXX",
    "product_name" : "str,
    "output_fields": ["company_entity", "product_name", "product_year", "product_month", "chapter", "file_name", "page_number","text"],
    "top_k": 5 -> default

  }'

    Returns:
        A dictionary containing the search results as a list of dictionaries with the following keys:
        "file_name": str,
        "id": int,
        "product_name": str,
        "product_year": int,
        "score": float,
        "text": str
    """
    try:
        data = request.json
        query = data["query"]
        collection_name = data["collection_name"]
        output_fields = ["text", "metadata", "page_number", "file_name", "product_name", "product_month",
                         "product_year", "chapter", "company_entity"]
        vector_field = data.get("vector_field", "vector")
        top_k = data.get("top_k", 12)
        # years = data.get("years", None)  # e.g. ["10.2014", "10.2023"] --> possibility of comparing more than two years...
        product_date = data.get("product_date", None)  # mm.yy
        product_name = data.get("product_name", None)
        chapter = data.get("chapter", None)
        company_entity = data.get("company_entity", None)

        any_field = data.get("any_field", None)  # --> "any" for example
        any_value = data.get("any_value", None)  # --> "a specific chapter name"

        connections.connect(
            host="102092af-5474-4a42-8dc2-35bb05ffdd0e.cvgfjtof0l91rq0joaj0.lakehouse.appdomain.cloud",
            port="31574",
            user=os.environ.get("MILVUS_USER"),
            password=os.environ.get("MILVUS_PASSWORD"),
            secure=True
        )

        emb_query = embed(query)
        collection = Collection(collection_name)
        collection.load()
        # print(collection.num_entities, flush=True)
        best_results = {}

        # if product_date:
        #     try:
        #         product_month, product_year = map(int, product_date.split("."))
        #     except ValueError:
        #         return jsonify({"error": "product_date must be MM.YYYY"}), 400

        product_month = product_year = None
        if product_date:
            result = find_most_recent_policy_before(product_date, collection, product_name)
            logging.info(result)
            print(result)
            if result:  # check if result is not None or empty
                product_year = result.get("product_year")
                product_month = result.get("product_month")
            else:
                logging.warning(
                    f"No policy found before or on {product_date} for product {product_name}. Using originally parsed date for filter if available.")

        # ── dynamic filter expression ─────────────────────────
        filter_expr = build_expr(
            product_name=product_name,
            product_month=product_month,
            product_year=product_year,
            company_entity=company_entity,
            chapter=chapter,
            **({any_field: any_value} if any_field and any_value else {})
        )

        hits = collection.search(
            data=[emb_query],
            anns_field=vector_field,
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )[0]
        # print(results)
        # output = [{"id": hit.id, "score": hit.distance, "text": hit.entity.get("text"), "file_name": hit.entity.get("file_name"), "product_year": hit.entity.get("product_year"), "product_name": hit.entity.get("product_name")} for hit in results[0]]

        results = [
            {
                "id": h.id,
                "score": h.distance,
                "text": h.entity.get("text"),
                "file_name": h.entity.get("file_name"),
                "product_name": h.entity.get("product_name"),
                "product_year": h.entity.get("product_year"),
                "product_month": h.entity.get("product_month"),
                "chapter": h.entity.get("chapter"),
                "page_number": h.entity.get("page_number"),
                "company_entity": h.entity.get("company_entity"),
            }
            for h in hits
        ]
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def build_expr(**kv):
    """
    Turn keyword arguments into a Milvus boolean-AND expression,
    skipping keys whose value is None/''.
    """
    fragments = []
    for k, v in kv.items():
        if v is None or v == "":
            continue
        # numeric vs. string fields
        if isinstance(v, (int, float)):
            fragments.append(f"{k} == {v}")
        else:
            fragments.append(f'{k} == "{v}"')
    expr = " && ".join(fragments) if fragments else ""
    logging.info(f"Filter expression: {expr}")
    return expr


@app.route("/compare", methods=["POST"])
def compare_definitions():
    try:
        data = request.json
        query = data["query"]
        collection_name = data["collection_name"]
        output_fields = data.get("output_fields")
        vector_field = data.get("vector_field", "vector")
        product_name = data["product_name"]
        topic_field = data.get("topic_field", None)  # --> "chapter" for example
        topic_value = data.get("topic_value", None)  # --> "a specific chapter name"
        years = data["years"]  # e.g. ["10.2014", "10.2023"] --> possibility of comparing more than two years...
        top_k = data.get("top_k", 3)

        connections.connect(
            host="102092af-5474-4a42-8dc2-35bb05ffdd0e.cvgfjtof0l91rq0joaj0.lakehouse.appdomain.cloud",
            port="31574",
            user=os.environ.get("MILVUS_USER"),
            password=os.environ.get("MILVUS_PASSWORD"),
            secure=True
        )

        collection_name = Collection(collection_name)
        collection_name.load()

        best_results = {}
        for year in years:
            result = find_most_recent_policy_before(year, collection_name, product_name)
            logging.info(result)
            if result:  # check if result is not None or empty
                best_results[year] = result

        response = compare_definitions_by_year(query, collection_name, vector_field, output_fields, product_name,
                                               topic_field, topic_value, years, best_results, top_k)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_product_names", methods=["GET"])
def get_product_names():
    """
    Retrieve all unique product_name values from a Milvus collection.
    Optional filter by company_entity query parameter.
    """
    try:
        collection_name = request.args.get("collection_name")
        company_entity = request.args.get("company_entity")

        if not collection_name:
            return jsonify({"error": "collection_name is required"}), 400

        connections.connect(
            host="102092af-5474-4a42-8dc2-35bb05ffdd0e.cvgfjtof0l91rq0joaj0.lakehouse.appdomain.cloud",
            port="31574",
            user=os.environ.get("MILVUS_USER"),
            password=os.environ.get("MILVUS_PASSWORD"),
            secure=True
        )
        coll = Collection(collection_name)
        coll.load()
        # build optional filter
        expr = f'company_entity == "{company_entity}"' if company_entity else ""

        # query product_name only
        docs = coll.query(expr=expr, limit=16000, output_fields=["product_name"])
        names = sorted({d.get("product_name") for d in docs if d.get("product_name")})
        return jsonify({"product_names": names}), 200
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

    collection = Collection("docling_helvetia")
    collection.load()

    result = find_most_recent_policy_before(date_str, collection, product_name)
    result["original_date"] = date_str
    return jsonify(result or {"error": "No match"})


def find_most_recent_policy_before(user_date_str, collection, product_name):
    user_date = datetime.strptime(user_date_str, "%m.%Y")
    # Dynamic parsing of product_name if available
    if product_name:
        filter_expr = build_expr(product_name=product_name)
    else:
        # Gets all docs before the user_date
        filter_expr = f"product_month<={user_date.month} && product_year<={user_date.year}"
    logging.info(f"most_recent_policy_before {user_date} - filter_expr: {filter_expr} for product_name: {product_name}")
    docs = collection.query(
        expr=filter_expr,
        limit=16000,
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


def compare_definitions_by_year(query, collection, vector_field, output_fields, product_name, topic_field,
                                topic_value, years, best_results, top_k=10):
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
    # collection = Collection(collection_name)
    query_vector = embed(query)
    for idx, year in enumerate(years, 1):
        logging.info(best_results[year])
        filter_parts = [f'product_year == {best_results[year]["product_year"]}',
                        f'product_month == {best_results[year]["product_month"]}', f'product_name == "{product_name}"']
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
            print(chunk_text)
            if chunk_text:
                text_chunks.append(chunk_text)
            # capture metadata once
            if not metadata_fields:
                for key in ["product_name",
                            "file_name"]:  # FOR ADDITIONAL META FIELDS: ["product_name", "product_year", "product_month", "file_name"]
                    value = hit.entity.get(key)
                    if value is not None:
                        metadata_fields[key] = value
        metadata_fields["orig_year"] = year
        metadata_fields["act_year"] = str(best_results[year]["product_month"]) + "." + str(
            best_results[year]["product_year"])
        results[f"context_{year}"] = "\n---\n".join(
            text_chunks)  # OLD: f"context_{best_results[year]['product_month']}.{best_results[year]['product_year']} Alternative: f"context_{idx}"
        results[f"metadata_{year}"] = metadata_fields  # OLD: f"metadata_{year}" Alternative: f"metadata_{idx}"

    return {
        "context": results
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
