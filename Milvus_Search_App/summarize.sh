curl -X POST http://127.0.0.1:5000/summarize_policy \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Was ist das Maximum der Limite?",
    "collection_name": "docling_helvetia",
    "filter": "product_name == \"Absicherungsplan\" && product_year == 2019",
    "output_fields": ["text"],
    "top_k": 5
  }'
