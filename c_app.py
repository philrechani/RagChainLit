from typing import List, Sequence
from typing_extensions import Annotated, TypedDict

from pathlib import Path

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

import chainlit as cl

from hug_rag_2 import Chug, VectorDatabase

chunk_size = 1000
chunk_overlap = 100

from config.CONFIG import MODEL_PATH, NAME

PDF_STORAGE_PATH = "./data/pdfs"

source = 'data\\pdfs\\DSM-5.pdf'

print('HELP')
config = {'context_length': 2048, 'gpu_layers': 50,'max_new_tokens': 1024,'stream':True}
model = Chug(model=MODEL_PATH,config=config)

print('help?')
def process_pdfs(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []  # type: List[Document]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for pdf_path in pdf_directory.glob("*.pdf"):
        print(f"Loading {pdf_path}...")
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {pdf_path}.")
        docs += text_splitter.split_documents(documents)

    print(f"Total documents after splitting: {len(docs)}")
    
    print("Creating Chroma index...")
    
    # change to the more fine-grained chroma storage
    return docs

def initialize_index(docs, client, collection, embedding_function):
    
    doc_search = Chroma(client=client,collection_name=collection.name, embedding_function=embedding_function)
    
    
    print("Chroma index created.")

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(namespace, db_url="sqlite:///record_manager_cache.sql")
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search

db = VectorDatabase(model)
db.initialize_vectorstore('persist')
results = db.query_collection(query_texts=[""],n_results=1,where={'source': source})
print(results)

docs = process_pdfs(PDF_STORAGE_PATH)
print(docs[0].metadata['source'])

if len(results['ids'][0]) == 0:
    db.add_to_collection(docs)
client, collection, embedding_function = db.chroma_objects
doc_search = initialize_index(docs, client, collection, embedding_function)    
print(doc_search)

print('HELP ME FOR THE LOVE OF GOD')

@cl.on_chat_start 
async def on_chat_start():
    
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    retriever = doc_search.as_retriever()
    
    def prompt_to_string(prompt_value):
        if isinstance(prompt_value, (dict, list)):
            return prompt_value
        return prompt_value.to_string()
    # Define the function that calls the model
        
    # test with this
    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | prompt_to_string
        | model
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )
                
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            print(token, end="", flush=True)
            return token
        
    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(),
            PostMessageHandler(msg)
        ]),
    ):
        await msg.stream_token(chunk)

    await msg.send()