@app.route("/get_summary", methods=["POST"])
def get_summary():
    try:
        data = request.json
        user_query = data["query"]  # Keep for Watsonx, not for Milvus
        collection_name = data["collection_name"]
        product_date = data["product_date"]
        product_name = data["product_name"]
        top_k = int(data.get("top_k", 10))

        # Validate required fields
        for field in ["query", "collection_name", "product_date", "product_name"]:
            if not data.get(field):
                return jsonify({"error": f"{field} is required"}), 400

        connections.connect(
            host="102092af-5474-4a42-8dc2-35bb05ffdd0e.cvgfjtof0l91rq0joaj0.lakehouse.appdomain.cloud",
            port="31574",
            user=os.environ.get("MILVUS_USER"),
            password=os.environ.get("MILVUS_PASSWORD"),
            secure=True
        )
        collection = Collection(collection_name)
        collection.load()

        # Build a valid filter expression
        filter_expr = build_expr(product_date=product_date, product_name=product_name)

        # Use query 
        results = collection.query(
            expr=filter_expr,
            output_fields=["text", "product_name", "product_date"],
            limit=top_k
        )

        result_list = []
        for doc in results:
            result_list.append({
                "product_name": doc.get("product_name"),
                "product_date": doc.get("product_date"),
                "text": doc.get("text")
            })

        return jsonify({"results": result_list, "count": len(result_list)}), 200

    except KeyError as ke:
        app.logger.error(f"Missing key in request: {ke}")
        return jsonify({"error": f"Missing required field: {ke}" }), 400
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
