import chainlit as cl
from config.CONFIG import MODEL_PATH

from pathlib import Path

from hug_rag import Chug, VectorDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.prompts import ChatPromptTemplate

chunk_size = 1000
chunk_overlap = 100

# === Functions ===
def process_pdfs(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = [] 
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

def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

def prompt_to_string(prompt_value):
        if isinstance(prompt_value, (dict, list)):
            return prompt_value
        return prompt_value.to_string()

# === Model Init ===
config = {'context_length': 2048, 'gpu_layers': 50, 'stream': True}
model = Chug(model_path=MODEL_PATH,config=config)

# === Connect to Database
source = 'data\\pdfs\\DSM-5.pdf'
PDF_STORAGE_PATH = "./data/pdfs"
db = VectorDatabase(model)
db.initialize_vectorstore('persist')

# === Check for current document ===
docs = process_pdfs(PDF_STORAGE_PATH) #get from the database instead if found
results = db.query_collection(query_texts=[""],n_results=1,where={'source': source})
if len(results['ids'][0]) == 0:
    db.add_to_collection(docs)
client, collection, embedding_function = db.chroma_objects
doc_search = initialize_index(docs, client, collection, embedding_function)

retriever = doc_search.as_retriever()    

template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    
prompt = ChatPromptTemplate.from_template(template)

# === Langchain Handler ===

from typing import Sequence

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, trim_messages
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.schema import StrOutputParser

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=State)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def call_model(state: State):
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | prompt_to_string
        | model
        | StrOutputParser()
    )
    trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke({"messages": trimmed_messages, "language": state["language"]})
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set('app',app)
    
@cl.on_message
async def on_message(message: cl.Message):
    app = cl.user_session.get('app')
    
    config = {"configurable": {"thread_id": "abc678"}}
    language = "English"
    print(message.content)
    input_message = [HumanMessage(message.content)]
    response = app.invoke({"messages": input_message, "language": language},
    config,)
    print(response)
    msg = cl.Message(content=response)
    
    await msg.send()