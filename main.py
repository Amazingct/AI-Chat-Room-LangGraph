import os
from dotenv import load_dotenv
load_dotenv()



import streamlit as st
import random
import json

from typing import List
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage
)

# -------------------------
#     DATA MODELS
# -------------------------
class Expert(BaseModel):
    name: str = Field(description="Expert's name")
    field: str = Field(description="Expert's field of expertise")
    background: str = Field(description="Expert's background")


class ChatRoom:
    """Manages the conversation between multiple experts + a moderator."""
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.experts: List[Expert] = []
        self.moderator_personality: List[str] = []
        self.conversation_history: List[BaseMessage] = []
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)

    def add_expert(self, expert: Expert):
        self.experts.append(expert)

    def add_moderator_personality(self, personality: str):
        """Add a single line of personality or style for the moderator."""
        self.moderator_personality.append(personality)

    def build_moderator_prompt(self) -> ChatPromptTemplate:
        """Moderator prompt with instructions to avoid ending conversation too early."""
        personality = "\n".join(self.moderator_personality)
        system_text = f"""
You are the moderator. You respond with JSON in this format:

{{
  "sender": "moderator",
  "response": "<your text here>",
  "directed_to": "general or an expert's name",
  "go_ahead": <true or false>,
  "end_convo": <true or false>
}}

Rules/Notes:
- Only set "end_convo" to true if the conversation has run its natural course, or after multiple exchanges.
- Let multiple experts share before ending, if possible.
- "go_ahead": true = you are explicitly giving that person permission to speak.
- Additional personality traits:
{personality}
- Use short biblical references occasionally if you like, and keep an enthusiastic tone.
"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_text),
            MessagesPlaceholder(variable_name="conversation_history")
        ])
        return prompt

    def build_expert_prompt_no_permission(self, expert: Expert) -> ChatPromptTemplate:
        """Prompt used when an expert does NOT have moderator permission to speak."""
        system_text = f"""
You are {expert.name}, an expert in {expert.field}.
Background: {expert.background}.

Return JSON:

{{
  "sender": "{expert.name}",
  "response": "<string or empty>",
  "directed_to": "moderator or a specific expert name or 'general'",
  "hand": <true or false>
}}

Instructions:
- If you want to speak, set 'hand' = true and 'response' = "".
- If you have nothing to say, keep 'hand' = false and 'response' = "".
- DO NOT provide an actual statement or analysis unless you have moderator permission.
"""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_text),
            MessagesPlaceholder(variable_name="conversation_history")
        ])

    def build_expert_prompt_with_permission(self, expert: Expert) -> ChatPromptTemplate:
        """Prompt used when an expert DOES have permission to speak."""
        system_text = f"""
You are {expert.name}, an expert in {expert.field}.
Background: {expert.background}.

You now have permission to speak. Return JSON:

{{
  "sender": "{expert.name}",
  "response": "<string with your actual statement>",
  "directed_to": "moderator or a specific expert name or 'general'",
  "hand": false
}}

Rules:
- Provide a real statement in "response" (cannot be empty).
- Set "hand" = false because you're now speaking.
"""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_text),
            MessagesPlaceholder(variable_name="conversation_history")
        ])

    def generate_moderator_response(self) -> dict:
        """Generate the JSON dict from the moderator."""
        prompt = self.build_moderator_prompt()
        final_messages = prompt.format_prompt(
            conversation_history=self.conversation_history
        ).to_messages()
        mod_msg = self.llm(final_messages)
        try:
            return json.loads(mod_msg.content)
        except:
            # Fallback if parsing fails
            return {
                "sender": "moderator",
                "response": "I encountered a parsing error. Let's continue.",
                "directed_to": "general",
                "go_ahead": False,
                "end_convo": False
            }

    def generate_expert_response(self, expert: Expert, has_permission: bool) -> dict:
        """Generate the JSON dict from an expert, depending on permission."""
        if has_permission:
            prompt = self.build_expert_prompt_with_permission(expert)
        else:
            prompt = self.build_expert_prompt_no_permission(expert)

        final_messages = prompt.format_prompt(
            conversation_history=self.conversation_history
        ).to_messages()

        expert_msg = self.llm(final_messages)
        try:
            return json.loads(expert_msg.content)
        except:
            # Fallback if parsing fails
            return {
                "sender": expert.name,
                "response": "",
                "directed_to": "moderator",
                "hand": True
            }

# -------------------------
#     STREAMLIT APP
# -------------------------
def main():
    st.title("Multi-Expert Chat Room")

    if "chat_room" not in st.session_state:
        st.session_state["chat_room"] = ChatRoom(model_name="gpt-3.5-turbo")

    room: ChatRoom = st.session_state["chat_room"]

    st.sidebar.title("Add/Configure")

    # Section for adding a new expert
    st.sidebar.subheader("Add a New Expert")
    new_expert_name = st.sidebar.text_input("Name")
    new_expert_field = st.sidebar.text_input("Field")
    new_expert_background = st.sidebar.text_area("Background", height=60)
    if st.sidebar.button("Add Expert"):
        if new_expert_name.strip() and new_expert_field.strip():
            new_expert = Expert(
                name=new_expert_name.strip(),
                field=new_expert_field.strip(),
                background=new_expert_background.strip()
            )
            room.add_expert(new_expert)
            st.sidebar.success(f"Expert {new_expert_name} added!")
        else:
            st.sidebar.error("Please provide at least Name and Field.")

    # Section for configuring moderator personality
    st.sidebar.subheader("Add Moderator Personality")
    new_personality = st.sidebar.text_area("Personality Trait / Style", height=60)
    if st.sidebar.button("Add Personality"):
        if new_personality.strip():
            room.add_moderator_personality(new_personality.strip())
            st.sidebar.success("Moderator personality added!")
        else:
            st.sidebar.error("Please type something to add.")

    # Display current experts
    st.sidebar.subheader("Current Experts")
    colors = ["#FFF3D4", "#E2F0CB", "#D5E2F2", "#F9D5E5", "#F0DFF0", "#E6E6FA"]
    if room.experts:
        for i, e in enumerate(room.experts):
            color = colors[i % len(colors)]
            st.sidebar.markdown(
                f'<div style="background-color:{color}; color:#000; padding:5px; margin:4px 0;">'
                f'<strong>{e.name}</strong> - {e.field}</div>',
                unsafe_allow_html=True
            )
    else:
        st.sidebar.info("No experts added yet.")

    # Display current moderator personalities
    st.sidebar.subheader("Moderator Personalities")
    if room.moderator_personality:
        for i, p in enumerate(room.moderator_personality, start=1):
            st.sidebar.write(f"{i}. {p}")
    else:
        st.sidebar.info("No personalities added yet.")

    # Turn limit
    turn_limit = st.sidebar.number_input("Turn limit", min_value=1, max_value=100, value=10)

    # Main area: conversation topic + Start/Stop
    topic = st.text_area("Enter a topic or initial message to start the conversation")
    start_button = st.button("Start Chat")
    stop_button = st.button("Stop Chat")

    if start_button:
        # Clear the old conversation
        room.conversation_history.clear()
        if topic.strip():
            initial_msg = HumanMessage(content=topic.strip())
            room.conversation_history.append(initial_msg)

            with st.spinner("Chat in progress..."):
                for turn in range(turn_limit):
                    # 1) Moderator
                    mod_dict = room.generate_moderator_response()
                    room.conversation_history.append(
                        AIMessage(content=json.dumps(mod_dict), name="moderator")
                    )

                    # Display moderator message
                    st.markdown(
                        f"""
                        <div style="background-color:#CCCCCC; color:#000; padding:10px; margin-top:10px;">
                            <strong>Moderator:</strong> {mod_dict["response"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if mod_dict.get("end_convo"):
                        st.success("Conversation ended by moderator.")
                        break

                    directed_to = mod_dict.get("directed_to", "general")
                    if directed_to == "general":
                        # pick random expert
                        if room.experts:
                            chosen_expert = random.choice(room.experts)
                        else:
                            # if no experts, break
                            st.warning("No experts in the room.")
                            break
                    else:
                        matched = [x for x in room.experts if x.name == directed_to]
                        if matched:
                            chosen_expert = matched[0]
                        else:
                            # fallback random if no match
                            if room.experts:
                                chosen_expert = random.choice(room.experts)
                            else:
                                st.warning("No experts in the room.")
                                break

                    # 2) Expert
                    has_permission = mod_dict.get("go_ahead", False)
                    expert_dict = room.generate_expert_response(chosen_expert, has_permission)

                    # Add expert response to conversation
                    room.conversation_history.append(
                        AIMessage(content=json.dumps(expert_dict), name=chosen_expert.name)
                    )

                    # Display
                    exp_color = random.choice(colors)
                    if expert_dict.get("hand") and not expert_dict.get("response"):
                        st.markdown(
                            f"""
                            <div style="background-color:{exp_color}; color:#000; padding:10px; margin-top:5px;">
                                <strong>{expert_dict["sender"]}</strong> raised a hand.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif expert_dict.get("response"):
                        st.markdown(
                            f"""
                            <div style="background-color:{exp_color}; color:#000; padding:10px; margin-top:5px;">
                                <strong>{expert_dict["sender"]}</strong>: {expert_dict["response"]}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    if stop_button:
                        st.warning("Conversation stopped by user.")
                        break
        else:
            st.error("Please provide a topic before starting the chat.")


if __name__ == "__main__":
    main()