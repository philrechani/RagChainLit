from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from llama_cpp import llama

import chromadb

from config.CONFIG import PERSIST_DIRECTORY
from langchain_core.embeddings import Embeddings
from langchain.schema import Document

from chromadb.api.types import Documents, Embeddings, Images
from typing import Union, TypeVar, Protocol, Any, Dict, Iterator, List, Optional

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

class RAG(LLM):
    def __init__(self,*args, **kwargs):
        self.kwargs = kwargs
        self.args = args
        self.client = llama(*args,embedding=False,**kwargs)
        self.current_mode = 'chat'
        self.context_size = self.kwargs['n_ctx']
        
        self.chroma_persist_directory = PERSIST_DIRECTORY
        self.current_collection_name = 'default_peanut_mushroom_5'
        
        self.current_client = None
        self.current_collection = None
        
        self.embedding_function: EmbeddingFunction[Embeddable] = EmbeddingFunction(self)
        
    def reinitialize_model(self, mode):
        if mode == self.current_mode:
            return  # No need to reinitialize if already in the correct mode

        if mode == 'embed':
            self.kwargs['embedding'] = True
            self.kwargs['vocab_only'] = False
            self.kwargs['n_ctx'] = 1  # Set context window to 1 for embeddings
        elif mode == 'chat':
            self.kwargs['embedding'] = False
            self.kwargs['vocab_only'] = False
            self.kwargs['n_ctx'] = self.context_size # Set appropriate context window for chat
        elif mode == 'tokenize':
            self.kwargs['embedding'] = False
            self.kwargs['vocab_only'] = True
            self.kwargs['n_ctx'] = self.context_size
            
        del self.client
        self.client = llama(*self.args,**self.kwargs)
        self.current_mode = mode    
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            # If streaming is enabled, we use the stream
            # method that yields as they are generated
            # and return the combined strings from the first choices's text:
            combined_text_output = ""
            for chunk in self._stream(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                combined_text_output += chunk.text
            return combined_text_output
        else:
            result = self.client(prompt=prompt, **kwargs)
            return result["choices"][0]["text"]
        
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens being generated.

        Yields:
            A dictionary like objects containing a string token and metadata.
            See llama-cpp-python docs and below for more.

        Example:
            .. code-block:: python

                from langchain_community.llms import LlamaCpp
                llm = LlamaCpp(
                    model_path="/path/to/local/model.bin",
                    temperature = 0.5
                )
                for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                        stop=["'","\n"]):
                    result = chunk["choices"][0]
                    print(result["text"], end='', flush=True)  # noqa: T201

        """
        result = self.client(prompt=prompt, stream=True, **kwargs)
        for part in result:
            logprobs = part["choices"][0].get("logprobs", None)
            chunk = GenerationChunk(
                text=part["choices"][0]["text"],
                generation_info={"logprobs": logprobs},
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    token=chunk.text, verbose=self.verbose, log_probs=logprobs
                )
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.kwargs['model_path'],
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"
        
    def embed(self, text, debug=True):
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

    def get_num_tokens(self, text: str) -> int:
        self.reinitialize_model('tokenize')
        tokenized_text = self.client.tokenize(text.encode("utf-8"))
        return len(tokenized_text)
