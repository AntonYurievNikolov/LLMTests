from langchain.vectorstores import Pinecone
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain import LLMChain
import gradio as gr
import pinecone
import os
import json



OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = OpenAIEmbeddings(
                                openai_api_key=OPENAI_API_KEY,
                                )
# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "esoteric"

vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

qa = RetrievalQA.from_chain_type(
# qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
                temperature=0.8,
                model_name="gpt-3.5-turbo",
                top_p= 0.7,
                frequency_penalty= 0.2,
                presence_penalty= 0.2
                ),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

#Defining the Tools
tools = [
                Tool(
                    name="In index of esoteric correspondences  about Tarot.",
                    func=qa.run,
                    description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                )
            ]

prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
            You have access to a single tool:"""
suffix = """Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo"
    ),
    prompt=prompt,
)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)


chat_history = []
vectordbkwargs = {"search_distance": 0.9}
def ask_question(query):
    answer = agent_chain.run(query)
    chat_history.append((query, answer))
    return answer, chat_history

def save_chat_history():
    os.makedirs("DataToIngest", exist_ok=True)
    base_file_name = f"chat_history_{len(os.listdir('DataToIngest')) // 2 + 1}"
    json_file_name = f"{base_file_name}.json"
    txt_file_name = f"{base_file_name}.txt"
    json_file_path = os.path.join("DataToIngest", json_file_name)
    txt_file_path = os.path.join("DataToIngest", txt_file_name)

    # Save as JSON
    with open(json_file_path, "w") as f:
        json.dump(chat_history, f)

    # Save as text
    with open(txt_file_path, "w") as f:
        for question, answer in chat_history:
            f.write(f"Q: {question}\nA: {answer}\n\n")

    return f"Chat history saved as {json_file_name} and {txt_file_name}"
# Gradio UI
demo = gr.Blocks()

with demo:
    textQ = gr.Textbox(label='Your Question:')
    textHistory = gr.Textbox(lines=6, label='History so far:')
    labelA = gr.Textbox(lines=3, label='Current Answer:')
    labelStatus = gr.Label(lines=1, label='Status')
    b1 = gr.Button("Ask Question")
    b2 = gr.Button("Save Conversation")

    b1.click(ask_question, inputs=textQ, outputs=[labelA,textHistory])
    b2.click(save_chat_history,  outputs=labelStatus)
    title = "Pinecone Q&A"
demo.launch(share=False)


