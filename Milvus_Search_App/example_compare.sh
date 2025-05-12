curl -X POST http://127.0.0.1:5000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Wie wird ein Unfall definiert?",
    "collection_name": "docling_helvetia",
    "product_name": "Absicherungsplan",
    "years": [2019, 2023],
    "output_fields": ["text", "product_name", "product_year", "file_name", "chapter", "page_number"],
    "vector_field": "vector",
    "top_k": 3
  }'

