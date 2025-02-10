import os
import streamlit as st
import random
import random
import getpass
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
load_dotenv()
from langchain.output_parsers import PydanticOutputParser
import os, time
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph
import json

import functools
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

# %%
class AgentState(TypedDict):
    chat_history: list[BaseMessage]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_config:dict

class Expert(BaseModel):
    name: str = Field(description="Expert's name")
    field: str = Field(description="Expert's field of expertise")
    background: str = Field(description="Expert's background", default="")

class ModeratorOutputObject(BaseModel):
    response: str = Field(description="Your contribution or respons)")
    sender: str = Field(description="Your name; always set this to moderator", default="moderator")
    directed_to: str = Field(description="If your response is directed to anyone specific in the room, your specify that here using their name else set this to general", default="general")
    go_ahead: bool = Field(description="Set to true if you want the expert to speak (This is for the moderator only)", default=False)
    end_convo: bool= Field(description="set to true if you feel the conversation has ended, only do this when you think all has been said", default=False)
    def to_json(self):
        return json.dumps(self.dict())
    
    
class OutputObject(BaseModel):
    response: str = Field(description="your contribution or response; this should be empty when you are raising your hand(s)")
    hand: bool = Field(description="set to true if your hand is raised else false", default=False)
    sender: str = Field(description="your name")
    directed_to: str = Field(description="if your response is directed to anyone specific in the room, your specify that here using their name else set this to general", default="general")

    def to_json(self):
        return json.dumps(self.dict())

# Set up a parser 
pydantic_parser = PydanticOutputParser(pydantic_object=OutputObject)
format_instructions = pydantic_parser.get_format_instructions()

# %%

class ChatRoom:
    def __init__(self):
        self.experts: List[Expert] = []
        self.llm = ChatOpenAI(model="gpt-4-1106-preview", model_kwargs = {"response_format":{"type": "json_object"}})
        self.workflow = StateGraph(AgentState)
        self.moderator_node = None
        self.recursion_limit = 20
        self.start_message = None
        self.graph = None
        self.all_participants = None
        self.moderator_extra_behaviours = ["be jovial and use puns"]
        

    def add_moderator_personality(self, behaviour: str):
        self.moderator_extra_behaviours.append(behaviour)
        
        
    def add_expert(self, expert: Expert):
        self.experts.append(expert)
        
   

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
                    "Here is the format for every single message you send: {response_format}."
                    "Here is the complete list of experts in the room {experts}"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(avatar=avatar)
        prompt = prompt.partial(expert_in=expert_in)
        prompt = prompt.partial(background=background)
        prompt = prompt.partial(response_format=format_instructions)
        e = [expert.dict() for expert in self.experts]
        e.append({"name":"moderator","field": "moderating the conversation"})
        prompt = prompt.partial(experts=e)
        return prompt | llm

    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        if isinstance(result, FunctionMessage):
            pass
        else:
            result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
        }
        
        
    def chat_init(self):
        # CREATE MODERATOR
        pydantic_parser = PydanticOutputParser(pydantic_object=ModeratorOutputObject)
        moderator_format_instructions = pydantic_parser.get_format_instructions()
        e = [expert.dict() for expert in self.experts]
        e.append({"name":"moderator","field": "moderating the conversation", "background":"host of event"})
        self.all_participants = e
        def mod(llm=self.llm, avatar="moderator",expert_in="organising conversations", tools=[], experts=e, response_format=moderator_format_instructions, system_message: str=None):
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
                        "Here is the format for every single message you send: {response_format}. "
                        "Here is the complete list of experts in the room {experts}. "
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            prompt = prompt.partial(avatar=avatar)
            prompt = prompt.partial(expert_in=expert_in)
            prompt = prompt.partial(response_format=response_format)
            prompt = prompt.partial(experts=experts)
            prompt = prompt.partial(personalities=", ".join(self.moderator_extra_behaviours))
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
            expert_agent = self.create_agent(self.llm, expert.name, expert.field, expert.background)
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
brain_surgery_expert = Expert(name="Dr. Brain", field="Brain Surgery", background="Neurosurgeon with 20 years of experience")
biotronics_expert = Expert(name="Dr. Biotronic", field="Biotronics", background="Pioneer in the field of Biotronics with numerous patents")
cardiology_expert = Expert(name="Dr. Heart", field="Cardiology", background="Cardiologist with 15 years of experience")
robotics_expert = Expert(name="Dr. Robot", field="Robotics", background="Robotics engineer with numerous inventions")
psychology_expert = Expert(name="Dr. Mind", field="Psychology", background="Psychologist with a focus on cognitive behavior")
genetics_expert = Expert(name="Dr. Gene", field="Genetics", background="Geneticist specializing in gene editing")
astrophysics_expert = Expert(name="Dr. Star", field="Astrophysics", background="Astrophysicist studying black holes")
nanotechnology_expert = Expert(name="Dr. Nano", field="Nanotechnology", background="Nanotechnologist working on nano robots")
quantum_physics_expert = Expert(name="Dr. Quantum", field="Quantum Physics", background="Quantum physicist working on quantum computing")
marine_biology_expert = Expert(name="Dr. Ocean", field="Marine Biology", background="Marine biologist studying deep sea creatures")

room.add_expert(brain_surgery_expert)
room.add_expert(biotronics_expert)
room.add_expert(cardiology_expert)
room.add_expert(robotics_expert)
room.add_expert(psychology_expert)
room.add_expert(genetics_expert)
room.add_expert(astrophysics_expert)
room.add_expert(nanotechnology_expert)
room.add_expert(quantum_physics_expert)
room.add_expert(marine_biology_expert)

room.add_moderator_personality("ensure everyone talks, quote bible in all response")
room.chat_init()

#MAIN APP
st.title('Chat Room')
st.sidebar.title('Chat Room Settings')

# Define colors
colors = ["red", "blue", "black"] * (len(room.all_participants) // 3 + 1)

# Display participants
st.sidebar.subheader('Participants')
for i, expert in enumerate(room.all_participants):
    color = colors[i]
    st.sidebar.markdown(f'<div style="background-color:{color};padding:10px;"><h3>{expert["name"]} - {expert["field"]}</h3></div>', unsafe_allow_html=True)

room.recursion_limit = st.sidebar.slider('Recursion limit', min_value=1, max_value=200, value=100)
start_button = st.sidebar.button('Start Chat')
stop_button = st.sidebar.button('Stop Chat')

start_message = {
  "topic": st.text_area("Enter a topic to start the conversation", height=200),
}

# No changes as we are sticking to original colors

if start_button:
    room.start_message = start_message
    with st.spinner('Chat in progress...'):
        for s in room.graph.stream(
            {
                "messages": [
                    HumanMessage(
                        content=f'''{room.start_message["topic"]}'''
                    )
                ],
            },
            # Maximum number of steps to take in the graph
            {"recursion_limit": room.recursion_limit},
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
            
            # Assign color to each participant
            experts_names = [expert["name"] for expert in room.all_participants if expert["name"] != "moderator"]
            color = colors[experts_names.index(name) if name in experts_names else -1]
            
            # Display message in a box with unique color
            st.markdown(f'<div style="background-color:{color};padding:10px;"><h3>{name}</h3>', unsafe_allow_html=True)
            if hand:
                st.markdown(f'<p><strong>Notification:</strong> {name} is raising his/her hands</p>', unsafe_allow_html=True)
            if content !="":
                st.markdown(f'<p>{content}</p></div>', unsafe_allow_html=True)
            
            if stop_button:
                st.success('Chat stopped.')
                break
