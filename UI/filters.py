import os
from pymilvus import connections, Collection
import streamlit as st

def get_product_names(company_entity=None):
    try:
        connections.connect(
            host=os.getenv("MILVUS_HOST"),
            port=os.getenv("MILVUS_PORT"),
            user=os.getenv("MILVUS_USER", "ibmlhapikey"),
            password=os.getenv("MILVUS_PASSWORD"),
            secure=True
        )
        # 1) get your LangChain store
        vc = st.session_state.vector_db

        # 2) grab the raw pymilvus Collection
        coll = Collection(vc.collection_name)
        coll.load()

        # 3) build your filter expression (optional)
        expr = f'company_entity == "{company_entity}"' if company_entity else ""

        # 4) query ONLY the product_name field
        rows = coll.query(
            expr=expr,
            output_fields=["product_name"],
            limit=16000,
        )

        # 5) dedupe & sort
        names = sorted({r["product_name"] for r in rows if r.get("product_name")})
    except Exception as e:
        st.error(f"Error retrieving product names: {e}")
        return []
    return names

def select_product_name(company_entity=None):
    product_names = get_product_names(company_entity=None)
    print(product_names)
    options = ["All"] + product_names
    selected_product = st.selectbox(
        "Choose the product",
        options=options,
        index=0,
    )
    # now store it for later use
    st.session_state.product_name = selected_product
    return selected_product