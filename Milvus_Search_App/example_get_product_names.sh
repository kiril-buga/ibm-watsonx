curl -X GET http://127.0.0.1:5000/get_product_names \ 
  -H "Content-Type: application/json" \
  -d '{
  "collection_name": "docling_helvetia",
  "product_name": "Absicherungsplan",
  }'