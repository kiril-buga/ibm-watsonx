curl -X POST http://127.0.0.1:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Wie wird ein Umfall definiert?",
    "collection_name": "docling_helvetia",
    "filter": "product_name == \"Absicherungsplan\" && product_year == 2019",
    "output_fields": ["company_entity", "product_name", "product_year", "chapter", "file_name", "page_number","text"],
    "top_k": 5

  }'
