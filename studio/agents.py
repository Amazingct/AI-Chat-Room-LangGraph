
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()
from langchain_core.messages import (
    AIMessage,

    FunctionMessage,
    HumanMessage,
)

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

search_function = TavilySearchResults(max_results=10)

tools = [search_function]
# Define LLM with bound tools
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful research assitant who use current information and data from internet search result to provide insights")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()


def math(state):

    result = graph.invoke(state)
    response = {"response":result['messages'][-1].content,
        "sender": "danny",
        "directed_to": "moderator",
        "hand": False
    }
    result = HumanMessage(json.dumps(response))
    return {
        "messages": [result],
    }



#MAPING
all_agents = {
    "danny" : math
}