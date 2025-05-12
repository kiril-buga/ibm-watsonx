curl -X POST http://127.0.0.1:5000/compare_by_date \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Vergleich der Vertragsbedingungen",
    "collection_name": "docling_helvetia",
    "product_name": "SEV Versicherungen",
    "dates": ["04.2011", "08.2009"],
    "output_fields": ["text", "file_name", "product_year", "product_month"],
    "vector_field": "vector",
    "top_k": 5
  }'
