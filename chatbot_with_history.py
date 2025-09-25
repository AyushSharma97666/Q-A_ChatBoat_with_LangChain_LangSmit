import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from dotenv import load_dotenv
import os
import uuid

# -----------------------
# Setup
# -----------------------
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Google Gemini"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------
# Initialize session state
# -----------------------
# Initialize sessions dict
if "sessions" not in st.session_state:
    st.session_state["sessions"] = {}

# Track the current session_id
if "current_session" not in st.session_state:
    st.session_state["current_session"] = None

# -----------------------
# Generate response
# -----------------------
def generate_response(modelName, question, max_tokens, memory):
    llm = ChatGoogleGenerativeAI(
        model=modelName,
        max_output_tokens=max_tokens,
    )

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user queries."),
            MessagesPlaceholder("chat_history"),
            ("user", "{question}")
        ]
    )

    # LLM Chain with session memory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    # Generate response
    return chain.predict(question=question)

# -----------------------
# Streamlit UI
# -----------------------
st.title("🧠 Q&A Chatbot with Session History")

st.sidebar.title("⚙️ Settings")

with st.sidebar:
    if st.button("➕ New Session"):
        session_id = str(uuid.uuid4())[:8]  # short unique id
        st.session_state["sessions"][session_id] = {
            "chat_history": [],  # store raw history
            "memory": ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        }
        st.session_state["current_session"] = session_id
        st.success(f"New session created: {session_id}")

    if st.session_state["sessions"]:
        session_id = st.selectbox(
            "Select Session", 
            options=list(st.session_state["sessions"].keys()), 
            index=list(st.session_state["sessions"].keys()).index(st.session_state["current_session"]) if st.session_state["current_session"] else 0
        )
        st.session_state["current_session"] = session_id
        st.info(f"Active session: {session_id}")

# Select the Google Gemini model
modelNames = st.sidebar.selectbox(
    "Select Google Gemini model",
    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0"]
)

# Adjust response parameter
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask any question 👇")

if st.session_state["current_session"] is not None:
    # Only run if a session is selected
    current_session = st.session_state["sessions"][st.session_state["current_session"]]
    memory = current_session["memory"]

    # Input box
    user_input = st.text_input("You:")

    if user_input:
        ai_reply = generate_response(modelNames, user_input, max_tokens,memory)
        current_session["chat_history"] = messages_to_dict(memory.chat_memory.messages)
        st.markdown(f"**🤖 Assistant:** {ai_reply}")

        
    # Restore stored messages if available
    if current_session["chat_history"]:
        memory.chat_memory.messages = messages_from_dict(current_session["chat_history"])

    # Show chat history
    if memory.chat_memory.messages:
        st.subheader("💬 Chat History")
        for msg in memory.chat_memory.messages:
            role = "🧑 You" if msg.type == "human" else "🤖 Assistant"
            st.write(f"**{role}:** {msg.content}")
else:
    st.warning("👉 Please create or select a session from the sidebar to start chatting.")