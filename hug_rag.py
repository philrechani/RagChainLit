from ctransformers import AutoModelForCausalLM
from config.CONFIG import MODEL_PATH, PERSIST_DIRECTORY

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

import chromadb

from langchain_core.embeddings import Embeddings
from langchain.schema import Document

from chromadb.api.types import Documents, Embeddings, Images
from typing import Union, TypeVar, Protocol, Any, Dict, Iterator, List, Optional

from pydantic import model_validator

Embeddable = Union[Documents,Images]
D = TypeVar("D", bound = Embeddable, contravariant = True)

class EmbeddingFunction(Protocol[D]):
    def __init__(self, chug_model):
        self.model = chug_model

    def __call__(self, input: D) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            embedding = self.model.embed(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, input: D) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            embedding = self.model.embed(text)
            embeddings.append(embedding)
        
        return embeddings[0]
    
    
class VectorDatabase:
    def __init__(self,model):
        self.chroma_persist_directory = PERSIST_DIRECTORY
        self._current_collection_name = 'default_peanut_mushroom_3'
        self._current_client = None
        self._current_collection = None
        self._embedding_function = EmbeddingFunction(model) #this should be the instantiated Chug class

    def initialize_vectorstore(self, type='memory', collection_name = None):
        if collection_name is None:
            collection_name = self._current_collection_name
            
        if not self._current_client:    
            if type == 'persist':
                self._current_client = chromadb.PersistentClient(path = self.chroma_persist_directory)
                
            if type == 'remote':
                self._current_client = chromadb.HttpClient()
                
            if type == 'memory':
                self._current_client = chromadb.Client()
            
            self._current_collection = self._current_client.get_or_create_collection(name = collection_name,embedding_function=self._embedding_function)
            
            # return self.current_client, self.current_collection
        else:
            print('Vector store already initialized')
        
    @property 
    def client(self):
        return self._current_client
    
    @property 
    def collection(self):
        return self._current_collection
    
    @property 
    def embedding_function(self):
        return self._embedding_function
    
    @property 
    def chroma_objects(self):
        return self._current_client, self._current_collection, self._embedding_function
    
    def add_to_collection(self,docs: list[Document], type = 'memory', collection_name = None):
        # this requires the langchain document object
        if collection_name is None:
            collection_name = self._current_collection_name
            
        if not self._current_client:    
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
        return self._current_collection.query(*args,**kwargs)
            
             
class Chug(LLM):
    client: Any #: :meta private:, will hold the LLM model object
    
    model_path: str
    """The path to a model file or directory or the name of a Hugging Face Hub
    model repo.""" 
    
    model_type: Optional[str] = None
    """The model type."""
    
    model_file: Optional[str] = None
    """The name of the model file in repo or directory."""
    
    config: Optional[Dict[str, Any]] = None
    """The config parameters."""
    
    lib: Optional[Any] = None
    """The path to a shared library or one of `avx2`, `avx`, `basic`."""
    
    def embed(self,text,**kwargs):
        return self.client.embed(text,**kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ctransformers"
    
    @model_validator(mode='before')
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate and load model from a local file or remote repo."""
        config = values.get("config") or {}
        values["client"] = AutoModelForCausalLM.from_pretrained(
            values.get("model_path"),
            model_type=values.get("model_type"),
            model_file=values.get("model_file"),
            lib=values.get("lib"),
            **config,
        )
        return values   
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if kwargs.get('stream'):
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
            return result
        
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:

        result = self.client(prompt=prompt, stream=True, **kwargs)
        for part in result:
            #logprobs = part["choices"][0].get("logprobs", None)
            chunk = GenerationChunk(
                text=part
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    token=chunk.text, verbose=self.verbose
                )
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_path,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"
