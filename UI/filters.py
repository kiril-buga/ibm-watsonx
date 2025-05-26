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

        # 4) query ONLY the product_name field in batches
        names: set[str] = set()
        iterator = coll.query_iterator(
            expr=expr,
            output_fields=["product_name"],
            batch_size=16000,
        )
        # iterate through batches
        while True:
            batch = iterator.next()
            if not batch:
                iterator.close()
                break
            # each batch is a list of dicts
            for record in batch:
                pname = record.get("product_name")
                if pname:
                    names.add(pname)
        names = list(names)
        # 5) dedupe & sort
    except Exception as e:
        st.error(f"Error retrieving product names: {e}")
        return []
    # convert set to sorted list for dropdown
    return sorted(names)

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
    
def get_product_dates(product_name=None):
    """
    Query Milvus for all distinct month.year combinations for a given product.
    """
    try:
        connections.connect(
            host=os.getenv("MILVUS_HOST"),
            port=os.getenv("MILVUS_PORT"),
            user=os.getenv("MILVUS_USER", "ibmlhapikey"),
            password=os.getenv("MILVUS_PASSWORD"),
            secure=True
        )
        coll = Collection(st.session_state.vector_db.collection_name)
        coll.load()
        # filter by product_name if provided
        expr = f'product_name == "{product_name}"' if product_name and product_name != "All" else ""
        rows = coll.query(
            expr=expr,
            output_fields=["product_month", "product_year"],
            limit=16000,
        )
        # build unique month.year strings
        dates = sorted({f"{r['product_month']}.{r['product_year']}" for r in rows
                        if r.get('product_month') is not None and r.get('product_year') is not None})
    except Exception as e:
        st.error(f"Error retrieving dates: {e}")
        return []
    return dates

def select_product_date():
    """
    Display a single dropdown of month.year values for the selected product.
    """
    pname = st.session_state.get("product_name")
    date_list = get_product_dates(product_name=pname)
    options = ["All"] + date_list
    selected = st.selectbox(
        "Choose product date",
        options=options,
        index=0,
    )
    st.session_state.product_date = None if selected == "All" else selected
    return selected