import logging
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

# import feedback
from full_chain import create_full_chain, ask_question
from vector_store import load_vector_db


st.set_page_config(page_title="RepoChat", page_icon="🦜️️🛠️", layout="wide")

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
    pass
    # Show sources
    # with st.expander("Sources", icon='📄', expanded=False):
    #     for idx, doc in enumerate(message["context"], 1):
            # print(doc["metadata"])
            # metadata = doc["metadata"]["dl_meta"]
            # filename = metadata.get("origin", {}).get("filename", "Unknown")
            # page_num = metadata.get("doc_items", {})[0].get("prov", {})[0].get("page_no")
            # headings = metadata.get("headings", "No headings available")
            # ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num} - headings: {headings}*]"
            # with st.popover(ref_title):
            #     st.caption(doc["page_content"])


@st.cache_resource
def get_retriever():
    # docs = load_pdf_files()
    # ensemble_retriever_from_docs(docs, embeddings=embeddings)
    vector_db = st.session_state.vector_db
    print(f"Vector DB {vector_db.collection_name}")
    return vector_db.as_retriever(search_type="similarity_score_threshold",
                                  search_kwargs={"score_threshold": 0.5, "k": 10}, )


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


def show_sidebar():
    with st.sidebar:
        # is_clicked = st.button("Clear Chat", on_click=local_storage.reset_chat_history)
        # if is_clicked:
        #     st.success("Chat history cleared! Please reload the page to start a new chat.")

        st.header("RepoChat Sources:")
        # Check if the vector db is loaded
        is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)

        if "rag_sources" not in st.session_state:
            vector_db = st.session_state.vector_db
            st.session_state.rag_sources = []
            # st.session_state.rag_sources = retrieve_sources_names(vector_db)

        # File upload input for RAG with documents
        # st.file_uploader(
        #     "📄 Upload a document",
        #     type=["pdf", "txt", "docx", "md"],
        #     accept_multiple_files=True,
        #     # on_change=load_doc_to_db,
        #     key="rag_docs",
        # )

        with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])


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

        chain = get_chain(model="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",)  # "llama-3.3-70B"

        # Initialize the cookie controller
        st.subheader("Ask me questions about reports")
        show_ui(chain,
                "Welche Frage hast du zu den Versicherungsdokumenten bei der Helvetia?")
        show_sidebar()
    else:
        st.stop()

run()