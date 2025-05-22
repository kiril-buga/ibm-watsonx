curl -X POST http://127.0.0.1:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Wie wird ein Umfall definiert?",
    "collection_name": "docling_helvetia",
    "product_date": "5.2019",
    "product_name" : "Absicherungsplan",
    "filter": "product_name == \"Absicherungsplan\" && product_year == 2019",
    "top_k": 12

  }'
