from langchain_community.llms import LlamaCpp

import chromadb

from config.CONFIG import PERSIST_DIRECTORY
from langchain_core.embeddings import Embeddings
from langchain.schema import Document

from chromadb.api.types import Documents, Embeddings, Images
from typing import Union, TypeVar, Protocol

#due to the split of the embedding model and the chat model, I will have to reorganize this class to manage the embedding model and chat model separately and remove them from memory when needed

#the del statement is the most guaranteed way of "garbage collecting"

Embeddable = Union[Documents,Images]
D = TypeVar("D", bound = Embeddable, contravariant = True)

class EmbeddingFunction(Protocol[D]):
    def __init__(self, rag_instance):
        self.rag = rag_instance

    def __call__(self, input: D) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            self.rag.reinitialize_model('embed')
            embedding = self.rag.embed(text, debug=False)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, input: D) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            self.rag.reinitialize_model('embed')
            embedding = self.rag.embed(text, debug=False)
            embeddings.append(embedding)
        
        return embeddings[0]

class RAG(LlamaCpp):
    def __init__(self, *args, n_ctx, vocab_only, embedding, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_only = vocab_only
        self.embedding = embedding
        self.n_ctx = n_ctx #init parameter
        self.context_size = n_ctx #chat parameter
        
        self.embedding_function: EmbeddingFunction[Embeddable] = EmbeddingFunction(self)
        
        self.current_mode = 'chat'
        
        self.chroma_host = ''
        self.chroma_persist_directory = PERSIST_DIRECTORY
        self.current_collection_name = 'default_peanut_mushroom_3'
        self.current_client = None
        self.current_collection = None
        
    def reinitialize_model(self, mode):
        if mode == self.current_mode:
            return  # No need to reinitialize if already in the correct mode

        if mode == 'embed':
            self.embedding= True
            self.vocab_only = False
            self.n_ctx = 1  # Set context window to 1 for embeddings
        elif mode == 'chat':
            self.embedding = False
            self.vocab_only = False
            self.n_ctx = self.context_size # Set appropriate context window for chat
        elif mode == 'tokenize':
            self.embedding = False
            self.vocab_only = True
            self.n_ctx = self.context_size
            
        super().__init__(*self.args, self.n_ctx, self.vocab_only, self.embedding,**self.kwargs)
        self.current_mode = mode    
        
    def chat(self, prompt,**kwargs):
        self.reinitialize_model('chat')
        return super().__call__(prompt,max_tokens=None,**kwargs) 
    
    def __call__(self, prompt,**kwargs):
        self.reinitialize_model('chat')
        return super().__call__(prompt,max_tokens= None,**kwargs)['choices'][0]['text']
    
    def create_chat_completion(self,*args, **kwargs):
        self.reinitialize_model('chat')
        #no longer valid
        return super().create_chat_completion(*args, **kwargs)
    
    def embed(self, text, debug=True):
        #how do?
        self.reinitialize_model('embed')
        if debug:
            print('Embedding:',text)
        return super().embed(text)[0]
    
    def initialize_vectorstore(self, type='memory', collection_name = None):
        if collection_name is None:
            collection_name = self.current_collection_name
            
        if not self.current_client:    
            if type == 'persist':
                self.current_client = chromadb.PersistentClient(path = self.chroma_persist_directory)
                
            if type == 'remote':
                self.current_client = chromadb.HttpClient()
                
            if type == 'memory':
                self.current_client = chromadb.Client()
            
            self.current_collection = self.current_client.get_or_create_collection(name = collection_name,embedding_function=self.embedding_function)
            
            return self.current_client, self.current_collection
        else:
            print('Vector store already initialized')
            
    def get_client(self):
        return self.current_client
    
    def get_collection(self):
        return self.current_collection
    
    def get_embedding_function(self):
        return self.embedding_function
    
    def get_chroma_objects(self):
        return self.current_client, self.current_collection, self.embedding_function
    
    def add_to_collection(self,docs: list[Document], type = 'memory', collection_name = None):
        # this requires the langchain document object
        if collection_name is None:
            collection_name = self.current_collection_name
            
        if not self.current_client:    
            self.initialize_vectorstore(type=type,collection_name=collection_name)
            
        texts = [doc.page_content for doc in docs]
        embeddings = [self.embed(text) for text in texts]
        dimension = len(embeddings[0])
        
        metadatas = [{**doc.metadata, 'dimension': dimension} for doc in docs]
        
        ids = [docs[i].metadata['source'] + f'_{i}' for i in range(len(docs))]
        
        self.current_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    def query_collection(self,*args,**kwargs):
        return self.current_collection.query(*args,**kwargs)
        