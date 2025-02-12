import streamlit as st
from langchain_core.messages import HumanMessage
import json

# Import the ChatRoom and Expert classes from your module
from studio.graph import room, Expert

st.title("Chat Room")
st.sidebar.title("Chat Room Settings")

# ========== 1) Manage Experts from the Sidebar ==========
with st.sidebar.expander("Manage Experts"):
    st.write("Add a new expert or remove an existing one.")
    # Add new expert
    new_expert_name = st.text_input("Expert Name", key="expert_name")
    new_expert_field = st.text_input("Expert Field", key="expert_field")
    new_expert_background = st.text_area("Expert Background", key="expert_background")

    if st.button("Add Expert"):
        if new_expert_name.strip():
            existing_experts = [e.name for e in room.get_experts_list()]
            if new_expert_name.strip() not in existing_experts:
                # Create the Expert object and add
                new_expert = Expert(
                    name=new_expert_name.strip(),
                    field=new_expert_field.strip(),
                    background=new_expert_background.strip()
                )
                room.add_expert(new_expert)
                st.success(f"Expert '{new_expert_name}' added!")
                st.rerun()
            else:
                st.warning("Expert name already exists. Please choose a different name.")
        else:
            st.warning("Please provide an Expert Name.")

    # Remove existing expert
    existing_experts = [e.name for e in room.get_experts_list()]
    if existing_experts:
        remove_expert_name = st.selectbox("Select Expert to remove", existing_experts)
        if st.button("Remove Expert"):
            room.remove_expert(remove_expert_name)
            st.success(f"Expert '{remove_expert_name}' removed!")
            st.rerun()
    else:
        st.info("No experts available to remove.")

# ========== 2) Manage Moderator Personalities ==========
with st.sidebar.expander("Manage Moderator Personalities"):
    st.write("Add or remove moderator personalities (descriptions).")
    new_personality = st.text_input("New Personality", key="new_personality")
    if st.button("Add Personality"):
        if new_personality.strip():
            room.add_moderator_personality(new_personality.strip())
            st.success(f"Moderator personality added: {new_personality}")
            st.rerun()
        else:
            st.warning("Please provide a personality description.")

    current_personalities = room.get_moderator_personality()
    if current_personalities:
        # Show them by index
        personality_index = st.selectbox(
            "Select personality to remove",
            range(len(current_personalities)),
            format_func=lambda i: current_personalities[i]
        )
        if st.button("Remove Personality"):
            room.remove_moderator_personality(personality_index)
            st.success("Removed personality!")
            st.rerun()
    else:
        st.info("No moderator personalities to remove.")

# ========== 3) Display Current Participants (Experts + Moderator) ==========
st.sidebar.subheader("Current Participants")
colors = ["red", "blue", "black", "green", "purple"] * 10  # add more if needed
for i, participant in enumerate(room.all_participants):
    color = colors[i]
    st.sidebar.markdown(
        f'<div style="background-color:{color};padding:10px;">'
        f"<h4>{participant['name']} - {participant['field']}</h4>"
        "</div>",
        unsafe_allow_html=True
    )

# ========== 4) Other Controls ==========
room.recursion_limit = st.sidebar.slider("Recursion limit", min_value=1, max_value=200, value=100)
start_button = st.sidebar.button("Start Chat")
stop_button = st.sidebar.button("Stop Chat")

# ========== 5) Chat Window ==========
start_message = {
    "topic": st.text_area("Enter a topic to start the conversation", height=200)
}

if start_button:
    room.start_message = start_message
    with st.spinner("Chat in progress..."):
        for s in room.graph.stream(
            {
                "messages": [
                    HumanMessage(content=f"{room.start_message['topic']}")
                ],
            },
            {"recursion_limit": room.recursion_limit},
        ):
            # s is a dictionary like {"moderator": {"messages": [HumanMessage(...)]}} etc.
            print(s)
            print("-----")

            name = list(s.keys())[0]
            try:
                parsed = json.loads(s[name]["messages"][0].content)
                content = parsed.get("response", "")
                hand = parsed.get("hand", False)
            except:
                content = "Chat ended"
                hand = False
                break

            # Assign color for experts vs. moderator
            # (We skip "moderator" in the color list indexing if you prefer.)
            participants_names = [p["name"] for p in room.all_participants]
            if name in participants_names:
                idx = participants_names.index(name)
                color = colors[idx]
            else:
                color = "lightgray"

            # Display message
            st.markdown(f'<div style="background-color:{color};padding:10px;"><h3>{name}</h3>', unsafe_allow_html=True)
            if hand:
                st.markdown(f'<p><strong>Notification:</strong> {name} is raising his/her hand!</p>', unsafe_allow_html=True)
            if content:
                st.markdown(f'<p>{content}</p></div>', unsafe_allow_html=True)

            if stop_button:
                st.success("Chat stopped.")
                break