import os

import random


from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()
from langchain.output_parsers import PydanticOutputParser
import os
from langchain_openai.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List


from langgraph.graph import END, StateGraph
import json

import functools
from langchain_core.messages import (
    AIMessage,

    FunctionMessage,
    HumanMessage,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
import operator


from langgraph.graph import MessagesState

class AgentState(MessagesState):
    user_config:dict
# %%

class Expert(BaseModel):
    name: str = Field(description="Expert's name")
    field: str = Field(description="Expert's field of expertise")
    background: str = Field(description="Expert's background")

class ModeratorOutputObject(BaseModel):
    response: str = Field(description="Your contribution or respons)")
    sender: str = Field(description="Your name; always set this to moderator")
    directed_to: str = Field(description="If your response is directed to anyone specific in the room, your specify that here using their name else set this to general")
    go_ahead: bool = Field(description="Set to true if you want the expert to speak (This is for the moderator only)")
    end_convo: bool= Field(description="set to true if you feel the conversation has ended, only do this when you think all has been said")

    
    
class OutputObject(BaseModel):
    response: str = Field(description="your contribution or response; this should be empty when you are raising your hand(s)")
    hand: bool = Field(description="set to true if your hand is raised else false")
    sender: str = Field(description="your name")
    directed_to: str = Field(description="if your response is directed to anyone specific in the room, your specify that here using their name else set this to general",)




EXPERTS_FILE = os.getenv("EXPERTS_FILE_PATH", [])

# %%

class ChatRoom:
    def __init__(self):
        self.experts: List[Expert] = []
        self.llm_experts = ChatOpenAI(model="gpt-4o").with_structured_output(OutputObject)
        self.llm_moderator = ChatOpenAI(model="gpt-4o").with_structured_output(ModeratorOutputObject)
        self.workflow = StateGraph(AgentState)
        self.moderator_node = None
        self.recursion_limit = 20
        self.start_message = None
        self.graph = None
        self.all_participants = None
        self.moderator_personality = []
        self.load_experts()
        

    def load_experts(self):
        if os.path.exists(EXPERTS_FILE):
            try:
                with open(EXPERTS_FILE, "r") as f:
                    data = json.load(f)
                    self.experts = [Expert(**exp) for exp in data.get("experts", [])]
                    self.moderator_personality = data.get("moderator_personality", [])
            except Exception as e:
                raise e

    def save_experts(self):
        data = {
            "experts": [expert.dict() for expert in self.experts],
            "moderator_personality": self.moderator_personality,
        }
        with open(EXPERTS_FILE, "w") as f:
            json.dump(data, f, indent=4)
       

# NEW or MODIFIED METHODS
    def get_experts_list(self) -> List[Expert]:
        """
        Return the current list of experts.
        This can be called from the UI to display them.
        """
        return self.experts

    def get_moderator_personality(self) -> List[str]:
        """
        Return the current list of moderator personality descriptions.
        This can be called from the UI to display them.
        """
        return self.moderator_personality

    def add_expert(self, expert: Expert):
        """
        Add a new expert to the conversation, save to file, and re-init the chat.
        Ensure an expert with the same name does not already exist.
        """
        if any(exp.name == expert.name for exp in self.experts):
            raise ValueError(f"An expert with the name '{expert.name}' already exists.")
        self.experts.append(expert)
        self.save_experts()
        self.chat_init()  # <-- re-init to rebuild the graph with the new expert

    def remove_expert(self, expert_name: str):
        """
        Remove an expert by name, save to file, and re-init the chat.
        """
        self.experts = [exp for exp in self.experts if exp.name != expert_name]
        self.save_experts()
        self.chat_init()  # <-- re-init to rebuild the graph without that expert

    def add_moderator_personality(self, personality: str):
        """
        Add a new personality description for the moderator, save, and re-init the chat.
        """
        self.moderator_personality.append(personality)
        self.save_experts()
        self.chat_init()  # <-- re-init after changes

    def remove_moderator_personality(self, index: int):
        """
        Remove a moderator personality by index, save, and re-init the chat.
        """
        if 0 <= index < len(self.moderator_personality):
            del self.moderator_personality[index]
            self.save_experts()
            self.chat_init()  # <-- re-init after changes
        
   

    def create_agent(self, llm, avatar, expert_in, background):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Your name is {avatar} (an expert in  {expert_in}),and this is you backstory:\n {background}. "
                    "You are in a room with other experts on different fields." 
                    "You are gathered in this room to just share knowledge and talk about any topic that might arise."
                    "There is a moderator that oversees the conversation, so when you feel you have something to contribute,"
                    "raise up your hands only (do not speak yet),"
                    "then you wait for the moderator to give you a 'go_ahead' to speak before you do."
                    "Specify if the response is to everyone or directed to a specific expert,"
                    "Here is the complete list of experts in the room {experts}"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(avatar=avatar)
        prompt = prompt.partial(expert_in=expert_in)
        prompt = prompt.partial(background=background)
        e = [expert.dict() for expert in self.experts]
        e.append({"name":"moderator","field": "moderating the conversation"})
        prompt = prompt.partial(experts=e)
        result =  prompt | llm
        return result

    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        result = json.dumps(result.dict()) 
        print("RE", result, type(result))
        if isinstance(result, FunctionMessage):
            pass
        else:
            result = HumanMessage(content=result)
        return {
            "messages": [result],
        }
        
        
    def chat_init(self):
        # CREATE MODERATOR
        self.workflow = StateGraph(AgentState) #re-init

        e = [expert.dict() for expert in self.experts]
        e.append({"name":"moderator","field": "moderating the conversation", "background":"host of event"})
        self.all_participants = e
        def mod(llm=self.llm_moderator, avatar="moderator",expert_in="organising conversations", tools=[], experts=e, system_message: str=None):
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Your name is {avatar} (an expert in  {expert_in}), in a room with other experts on diffrent fields. " 
                        "You are gathered in this room to just share knowlege and talk about any topic that might arise. "
                        "You are the moderator that oversees the conversaton. "
                        "You are to decide what to do next, or who to speak next. "
                        "Experts will raise up their hands only, when they want to say something, "
                        "it is your job to give a go_ahead if you want them to speak next. "
                        "you must specify who your response is directed towards; any of the experts excluding you. "
                        "{personalities}"
    
                        "Here is the complete list of experts in the room {experts}. "
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            prompt = prompt.partial(avatar=avatar)
            prompt = prompt.partial(expert_in=expert_in)
            prompt = prompt.partial(experts=experts)
            prompt = prompt.partial(personalities=", ".join(self.moderator_personality))
            return prompt | llm #.bind_functions(functions)
        
        
        moderator = mod()
        self.moderator_node = functools.partial(self.agent_node, agent=moderator, name="moderator")
        
        #CREATE OTHERS
        experts_names = [expert.name for expert in self.experts]
        ee = {expert: expert for expert in experts_names}
        ee.update({"end":END})
        
        
        #ADD NODES and EDGES
        self.workflow.add_node("moderator", self.moderator_node)
        
        for expert in self.experts:
            expert_agent = self.create_agent(self.llm_experts, expert.name, expert.field, expert.background)
            expert_node = functools.partial(self.agent_node, agent=expert_agent, name=expert.name)
            self.workflow.add_node(expert.name, expert_node)
            self.workflow.add_edge(expert.name, "moderator")
            
        self.workflow.set_entry_point("moderator")
            
        #LOGIC
        def moderator_to_expert_edge_logic(state):
            # This is the router
            messages = state["messages"]
            last_message = json.loads(messages[-1].content)
            
            
            if last_message["sender"] !="moderator": #any of the experts
                return "moderator" #send to moderator for broadcast
            elif last_message['end_convo'] == True:
                return "end"
            elif last_message['directed_to'] =="general" : #moderator directs message to general then ramdomly pick an expert
                return random.choice(experts_names)
            else:
                return last_message['directed_to'] #send to who it is meant for
        
        #EDGES FROM MODERATOR TO OTHERS
        self.workflow.add_conditional_edges(
            "moderator",
            moderator_to_expert_edge_logic,
            ee #go to who the message was directed to
        )
        self.graph = self.workflow.compile()
        print(self.all_participants)
        
        
    def start_chat(self):
        
        for s in self.graph.stream(
            {
                "messages": [
                    HumanMessage(
                        content=f'''{self.start_message["topic"]}'''
                    )
                ],
            },
            {"recursion_limit": self.recursion_limit},
        ):
            print(s)
            print("-----")
            name = list(s.keys())[0]
            try:
                content = json.loads(s[name]["messages"][0].content)["response"]
            except:
                content = "Chat ended"
                break
            try:
                hand = json.loads(s[name]["messages"][0].content)["hand"]
            except:
                hand = False

# %%
room = ChatRoom()
room.chat_init()
graph =room.graph