import os
import streamlit as st
from groq import Groq 
from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv("groq_api_key")

st.sidebar.title("Personalization")
prompt = st.sidebar.title("System Prompt")
model = st.sidebar.selectbox('Choose a model', ['groq/compound', 'groq/compound-mini'])

#Groq Client
client = Groq(api_key=groq_api_key)

#Streamlit Interface
st.title("Chat with GROQ LLM ")

#Initialise session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Enter your query: ", "")

if st.button("Submit"):
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
    
                {"role": "user", "content": user_input}
            ]
        )
        #Store the query and response in history
        response = chat_completion.choices[0].message.content
        st.session_state.history.append({"query" : user_input, "response" : response})

        #Display the response
        st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)

        #Display chat history
        st.sidebar.title("Chat History")
        for i, entry in enumerate(st.session_state.history):
            if st.sidebar.button(f"Query {i+1}: {entry['query']}"):
                st.markdown(f'<div class="response-box">{entry["response"]}</div>', unsafe_allow_html=True)