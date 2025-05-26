import logging
import os

import streamlit as st
from dotenv import load_dotenv

from filters import select_product_date, select_product_name

load_dotenv()
# from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

# import feedback
from full_chain import create_full_chain, ask_question
from vector_store import load_vector_db


st.set_page_config(page_title="Helvetia", page_icon="🦜️️🛠️", layout="wide")

def show_ui(qa, prompt_to_user="Wie kann ich Ihnen helfen?"):
    # How many messages to show by default
    DEFAULT_MAX_MESSAGES = 20
    # Use a session state variable to track how many messages to display.
    if "max_displayed_messages" not in st.session_state:
        st.session_state.max_displayed_messages = DEFAULT_MAX_MESSAGES

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]
        logging.info(f"Initial message {st.session_state.messages}")

    total_msgs = len(st.session_state.messages)
    # If there are more messages than we want to display,
    # offer a button to load older messages.
    if total_msgs > st.session_state.max_displayed_messages:
        if st.button("Load Previous Messages"):
            st.session_state.max_displayed_messages = total_msgs

    # Only render the last N messages.
    messages_to_render = st.session_state.messages[-st.session_state.max_displayed_messages:]

    # Display chat messages
    for message in messages_to_render:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Display feedback form only for assistant responses
            if message["role"] == "assistant" and message != st.session_state.messages[0]:
                display_context(message)
                # feedback.display_feedback(message)
    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        display_new_message(qa, prompt)
    st.caption("AI chatbots may produce errors. Always verify the information and consult the appropriate expert.")


@st.fragment
def display_new_message(qa, prompt):
    # Generate a response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt)
                # Update the message to store in cookies. Problem is in the context structure
                message = {
                    "role": "assistant",
                    "run_id": st.session_state.get("run_id"),
                    "content": response["answer"],
                    "context": [
                        {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                        }
                        for doc in response["context"]
                    ]
                }
                st.session_state.messages.append(message)

                display_context(message)
                # feedback.display_feedback(message)
                # local_storage.set_chat_history(st.session_state.messages)

@st.fragment
def display_context(message):
    # Show sources
    with st.expander("Sources", icon='📄', expanded=False):
        selected_product = st.session_state.get("product_name")
        selected_date = st.session_state.get("product_date")
        print(f"context: {message['context']}")
        for idx, doc in enumerate(message["context"], 1):
            print(doc)

            metadata = doc.get("metadata", {})
            filename = metadata.get("file_name", "Unknown")
            page_num = metadata.get("page_number", "Unknown")
            product_name = metadata.get("product_name", "Unknown Product")
            headings = metadata.get("chapter", "No headings available")
            product_year = metadata.get("product_year", "unknown")
            product_month = metadata.get("product_month", "unknown")
            company_entity = metadata.get("company_entity", "unknown")
            ref_title = f":blue[Reference {idx}: * Product: {product_name} - {filename} - Page.{page_num} - Chapter: {headings} - Date: {product_month}.{product_year} - Company: {company_entity}*]"
            with st.popover(ref_title):
                st.caption(doc["page_content"])


def get_retriever() -> "VectorStoreRetriever":
    """
    Build a Milvus retriever that really filters on
    product_name / product_month / product_year.
    """
    vdb = st.session_state.vector_db

    # ---------- 1. Build Milvus boolean expression -----------------
    parts = []
    pname = st.session_state.get("product_name")
    if pname and pname != "All":
        parts.append(f'product_name == "{pname}"')

    pdate = st.session_state.get("product_date")
    if pdate and pdate != "All":
        try:
            m, y = map(int, pdate.split("."))
            parts.append(f"product_month == {m}")
            parts.append(f"product_year == {y}")
        except ValueError:
            st.warning("Bad month.year in dropdown – skipping date filter.")

    expr = " && ".join(parts)     # Milvus uses && for AND

    # ---------- 2. Pack search kwargs --------------------------------
    # include output_fields so metadata is returned from Milvus
    skw = {
        "k": 12,
        "score_threshold": 0.5,
    }
    if expr:                      # only add when we have something
        skw["expr"] = expr

    #st.write(f"Milvus collection: {vdb.collection_name} | search_kwargs: {skw}")

    # ---------- 3. Build retriever -----------------------------------
    return vdb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=skw,
    )



def get_chain(model=None, api_key=None, huggingfacehub_api_token=None):
    retriever = get_retriever()
    chain = create_full_chain(
        model=model,
        api_key=api_key,
        retriever=retriever,
        huggingfacehub_api_token=huggingfacehub_api_token,
        # chat_memory=StreamlitChatMessageHistory(key="chat_history")
    )
    return chain


def get_secret_or_input(secret_key, secret_name, info_link=None):
    secret_value = os.getenv(secret_key)
    # if secret_key in st.secrets:
    #     st.write("Found %s secret" % secret_key)
    #     secret_value = st.secrets[secret_key]
    if secret_value:
        st.write("Found %s secret" % secret_key)
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


@st.cache_resource
def set_vector_db(huggingfacehub_api_token=None):
    model_kwargs = {'token': huggingfacehub_api_token, 'device': 'cpu', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large", model_kwargs=model_kwargs,
                                       encode_kwargs=encode_kwargs)
    return load_vector_db(embeddings, collection_name="docling_helvetia")

def run():
    ready = True

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # groq_api_key = st.session_state.get("GROQ_API_KEY")
    # cerebras_api_key = st.session_state.get("CEREBRAS_API_KEY")
    # huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    # with st.sidebar:
        # if not cerebras_api_key:
        #     with st.popover("🔐 Cerebras API Key"):
        #         cerebras_api_key = get_secret_or_input('CEREBRAS_API_KEY', "CEREBRAS_API_KEY",
        #                                                info_link="https://inference-docs.cerebras.ai/integrations")
        # if not huggingfacehub_api_token:
        #     with st.popover("🔐 HuggingFace API Key"):
        #         huggingfacehub_api_token = get_secret_or_input('HUGGINGFACEHUB_API_TOKEN', "HuggingFace Hub API Token",
        #                                                        info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

    # if not groq_api_key:
    #     st.warning("Missing GROQ_API_KEY")
    #     ready = False
    # if not cerebras_api_key:
    #     st.warning("Missing CEREBRAS_API_KEY")
    #     ready = False
    # if not huggingfacehub_api_token:
    #     st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
    #     ready = False
    # if not check_password():
    #     ready = False

    if ready:
        st.session_state.vector_db = set_vector_db()
        # local_storage.load_chat_history()
        select_product_name()
        select_product_date()

        chain = get_chain(model="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",)  # "llama-3.3-70B"

        # Initialize the cookie controller
        st.subheader("Ask me questions about Helvetia insurance documents")
        show_ui(chain,
                "Welche Frage hast du zu den Versicherungsdokumenten bei der Helvetia?")
    else:
        st.stop()

# start the app
run()