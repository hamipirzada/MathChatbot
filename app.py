import streamlit as st
from langchain.chains import LLMMathChain, LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Setup the streamlit app
st.set_page_config(page_title="Text To Math Problem Solver and Data Search Assistant")
st.title("Text To Math Problem Solver Using Google Gemma 2")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API key to continue...")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key= groq_api_key)

# Initialize the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = "wikipedia",
    func = wikipedia_wrapper.run,
    description = "A tool for searching the internet to find the various information on the topic"

)

# initialize the math tool
math_chain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name = "Calculator",
    func = math_chain.run,
    description = "A tool for performing mathematical calculations"
)

prompt = """You are an agent responsible for solving users mathematical questions. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question : {question}
Answer : 
"""

prompt_template = PromptTemplate(
    input_variables= ["question"],
    template= prompt
)

# Combine all the tools into chain
chain = LLMChain(llm = llm, prompt = prompt_template)

reasoning_tool = Tool(
    name = "Reasoning tool",
    func= chain.run,
    description= "A tool for answering logic-based and reasoning questions."

)

# initialize the agents
assistant_agent = initialize_agent(
    tools= [wikipedia_tool, calculator, reasoning_tool],
    llm= llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_errors = True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer all your maths questons."}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to generate the response
def generate_response(question):
    response = assistant_agent.invoke({'input': question})
    return response

# Lets start the interaction
question = st.text_area("Enter your question:", "18 men can reap a field in 35 days. For reaping the same field in 15 days, how many men are required?")
if st.button("Get the Answer"):
    if question:
        with st.spinner("Calculating ..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("...Response...")
            st.success(response)

    else:
        st.error("Please enter a question.")