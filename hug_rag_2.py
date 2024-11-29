from config.CONFIG import MODEL_PATH, PERSIST_DIRECTORY

from langchain_core.language_models.llms import LLM
from langchain_core.messages import BaseMessage

import chromadb

from langchain_core.embeddings import Embeddings
from langchain.schema import Document

from chromadb.api.types import Documents, Embeddings, Images
from typing import Union, TypeVar, Protocol, Any, Dict, Sequence, List, Optional, Iterator

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.utils import pre_init
from langchain_core.outputs import Generation, GenerationChunk, LLMResult

from functools import partial

from pydantic import model_validator

Embeddable = Union[Documents,Images]
D = TypeVar("D", bound = Embeddable, contravariant = True)

class EmbeddingFunction(Protocol[D]):
    def __init__(self, chug_model):
        self.model = chug_model

    def __call__(self, input: D) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        if isinstance(input, BaseMessage):
            input = [input.content]
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
        
        self._current_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    def query_collection(self,*args,**kwargs):
        return self._current_collection.query(*args,**kwargs)
            
             
class Chug(LLM):
    client: Any  #: :meta private:

    model: str
    """The path to a model file or directory or the name of a Hugging Face Hub
    model repo."""

    model_type: Optional[str] = None
    """The model type."""

    model_file: Optional[str] = None
    """The name of the model file in repo or directory."""

    config: Optional[Dict[str, Any]] = None
    """The config parameters.
    See https://github.com/marella/ctransformers#config"""

    lib: Optional[str] = None
    """The path to a shared library or one of `avx2`, `avx`, `basic`."""
    
    def embed(self,text,**kwargs):
        return self.client.embed(text,**kwargs)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "model_type": self.model_type,
            "model_file": self.model_file,
            "config": self.config,
        }
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ctransformers"
    
    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that ``ctransformers`` package is installed."""
        try:
            from ctransformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "Could not import `ctransformers` package. "
                "Please install it with `pip install ctransformers`"
            )

        config = values["config"] or {}
        values["client"] = AutoModelForCausalLM.from_pretrained(
            values["model"],
            model_type=values["model_type"],
            model_file=values["model_file"],
            lib=values["lib"],
            **config,
        )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            stop: A list of sequences to stop generation when encountered.

        Returns:
            The generated text.

        Example:
            .. code-block:: python

                response = llm.invoke("Tell me a joke.")
        """
        text = []
        _run_manager = run_manager or CallbackManagerForLLMRun.get_noop_manager()
       
        for chunk in self.client(prompt, stop=stop, stream=True):
            text.append(chunk)
            _run_manager.on_llm_new_token(chunk, verbose=self.verbose)
        return "".join(text)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous Call out to CTransformers generate method.
        Very helpful when streaming (like with websockets!)

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python
                response = llm.invoke("Once upon a time, ")
        """
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)

        text = ""
        for token in self.client(prompt, stop=stop, stream=True):
            if text_callback:
                await text_callback(token)
            text += token

        return text