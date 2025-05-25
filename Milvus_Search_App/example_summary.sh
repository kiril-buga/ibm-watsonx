curl -X POST http://127.0.0.1:5000/get_summary \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Was ist die Zusammenfassung?",
    "collection_name": "docling_helvetia",
    "product_date": "5.2019",
    "product_name" : "Absicherungsplan",
    "filter": "product_name == \"Absicherungsplan\" && product_year == 2019",
    "top_k": 12

  }'

curl -X POST http://192.168.1.116:5000/get_summary \
  -H "Content-Type: application/json" \
  -d '{
    "query": "koennen Sie das Dokument zusammenfassen?",
    "collection_name": "docling_helvetia",
    "product_date": "8.2019",
    "product_name" : "Absicherungsplan",
    "filter": "product_name == \"Absicherungsplan\"",
    "top_k": 12

  }'