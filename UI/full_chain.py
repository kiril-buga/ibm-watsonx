import logging
import os

import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
# from langchain_cerebras import ChatCerebras
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import collect_runs

# from basic_chain import get_model
# from filter import ensemble_retriever_from_docs
# from local_loader import load_txt_files
# from memory import create_memory_chain
# from rag_chain import make_rag_chain


# @traceable(run_type="chain")
def get_model(repo_id="test", **kwargs):
    logging.info(f"model:{repo_id}")
    # cerebras_api_key = kwargs.get("CEREBRAS_API_KEY")
    generate_params = {
        # keep generating until we really stop it
        GenParams.MAX_NEW_TOKENS: 1024,
        GenParams.STOP_SEQUENCES: [],  # ⚠️ turn off the default “\n\n”
        # optional but recommended
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.TEMPERATURE: 0,  # deterministic
    }

    creds = Credentials(
        api_key=os.getenv("API_KEY"),
        url="https://eu-de.ml.cloud.ibm.com",  # region root
    )
    model = ModelInference(
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        credentials=creds,
        project_id=os.getenv("PROJECT_ID"),
        params=generate_params
    )
    chat_model = WatsonxLLM(
        model=model
        # rate_limiter=rate_limiter,
    )
    # groq_api_token = kwargs.get("GROQ_API_KEY")
    # chat_model = ChatGroq(
    #     model=repo_id,
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=3,
    # )
    return chat_model


def create_full_chain(model, api_key, retriever, huggingfacehub_api_token=None, chat_memory=ChatMessageHistory()):
    model = get_model(model, huggingfacehub_api_token=huggingfacehub_api_token)  # "llama-3.3-8b-versatile"
    st.session_state.model = model  # Share the model with other functions
    logging.info(f"chat_memory: {chat_memory}")
    # history_aware_retriever = create_memory_chain(model, retriever, "", chat_memory)
    rag_chain = make_rag_chain(model, retriever, history_aware_retriever=None, chat_memory=chat_memory)
    return rag_chain


def make_rag_chain(model, retriever, history_aware_retriever, chat_memory):
    # We will use a prompt template from langchain hub.
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """You are a helpful assistant that answers questions about technical documents in any language. 
             Always respond in the language of the user prompt, ignore the language of the "Context".

             Only use the factual information from the document(s) to answer the question(s). Keep your answers concise and to the point.

             If you do not have have sufficient information to answer a question, politely refuse to answer and say that you don't know.
             \nRelevant documents are retrieved in the "Context" below.\n"""
             "Context: {context}"
             ),
            ("human", "{input}"),
            ("placeholder", "{chat_history}"),
        ])

    qa_chain = create_stuff_documents_chain(model, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain) # create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain

def ask_question(chain, query):
    response = ""
    placeholder = st.empty()

    # Helps to not exceed the token limit for context window
    # selected_messages = trim_messages(
    #     st.session_state.get("messages", []), # chat_history excluding the input message
    #     # Please see API reference for trim_messages for other ways to specify a token counter.
    #     token_counter=len,  # based on the current model like ChatOpenAI(model="gpt-4o"),
    #     max_tokens=10,  # <-- token limit
    #     # The start_on is specified
    #     # Most chat models expect that chat history starts with either:
    #     # (1) a HumanMessage or
    #     # (2) a SystemMessage followed by a HumanMessage
    #     # start_on="human" makes sure we produce a valid chat history
    #     start_on="human",
    #     # Usually, we want to keep the SystemMessage
    #     # if it's present in the original history.
    #     # The SystemMessage has special instructions for the model.
    #     include_system=True,
    #     strategy="last",
    # )

    with collect_runs() as cb:
        for token in chain.stream(
                {"input": query, "chat_history": st.session_state.get("messages", [])},
                config={"configurable": {"session_id": "any"}, "run_name": "repochat_chain", },
                # config={"configurable": {"session_id": "foo"}}
        ):
            response += token
            if "answer" in token:
                placeholder.markdown(response["answer"] + "▌")
        st.session_state.run_id = str(cb.traced_runs[0].id)
    placeholder.markdown(response["answer"])
    return response


def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    docs = load_txt_files()
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    chain = create_full_chain(ensemble_retriever)

    queries = [
        "Generate a grocery list for my family meal plan for the next week(following 7 days). Prefer local, in-season ingredients."
        "Create a list of estimated calorie counts and grams of carbohydrates for each meal."
    ]

    for query in queries:
        response = ask_question(chain, query)
        console.print(Markdown(response.content))


if __name__ == '__main__':
    # this is to quiet parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()