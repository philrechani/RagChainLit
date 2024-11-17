from langchain.chains.llm_math.base import LLMMathChain
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_community.chat_models import ChatOpenAI
from typing import *
from langchain.tools import BaseTool

import chainlit as cl
from chainlit.sync import run_sync


class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name: str= "human" 
    description: tuple = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query).send())
        return res["content"]

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        res = await cl.AskUserMessage(content=query).send()
        return res["output"]

from hug_rag import Chug, VectorDatabase
from config.CONFIG import MODEL_PATH

config = {'context_length': 2048, 'gpu_layers': 50}
model = Chug(model_path=MODEL_PATH,config=config)

@cl.on_chat_start
def start():
    llm = model
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        HumanInputChainlit(),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
            coroutine=llm_math_chain.arun,
        ),
    ]
    agent = initialize_agent(
        tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()