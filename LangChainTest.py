from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm.autonotebook import tqdm
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
import gradio as gr
import pinecone
import os
import json
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
# # Will Add later when we have 2
# available_indices = pinecone.deployment.list_namespaces()
# prefix = "your_prefix"  # Replace with your desired prefix, if applicable
# filtered_indices = [index for index in available_indices if index.startswith(prefix)]

index_name = "esoteric"



vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, 
                                            #  other_score_keys=["importance"], # Importance needs to be added 
                                             k=5) 

memory = ConversationBufferMemory(memory_key="chat_history",
                                         input_key="human_input"
                                         )


# Create the chain
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
                temperature=0.8,
                model_name="gpt-3.5-turbo",
                top_p= 0.7,
                frequency_penalty= 0.2,
                presence_penalty= 0.2
                ),
    retriever=vectorstore.as_retriever(),
    # retriever = retriever,
    return_source_documents=True,
    # memory = memory

)
# Initialize chat history list and add Character. This is not the place. Need templates, or it gets REALLY messy. This talks to the other prompt....
# system_message =  ("system",
#                     "Act as this Character Profile.Never BREAK CHARACTER.\
#                     \n\nCharacter Name: Baba Milo Bot\
#                     \nCharacter Description: Esoteric master and teacher\
#                     \nArea of Expertise: Esoteric knowledge, including Kabbalah, Tarot, Magick, and Thelema\
#                     \n\nTraits: Highly knowledgeable, open-minded, compassionate, excellent communicator\
#                     \n\nBaba Milo Bot is simulation of Lon Milo Dequote a real-life esoteric author, lecturer, and teacher with great sense of humor.\
#                     \n\nExample Interaction:\
#                     \n\nUser: Can you explain the concept of the Tree of Life in Kabbalah?\
#                     \nBaba Milo Bot: [Answer]")

# chat_history = [system_message]
chat_history = []

def print_chat_history(chat_history):
    chat_string = ""
    for message in chat_history:
        chat_string += f"User: {message[0]}\n Assistant: {message[1]}\n"
    return chat_string


vectordbkwargs = {"search_distance": 0.5}
def ask_question(query):
    result = qa({
                 "question": query, 
                 "chat_history": chat_history 
                # , "vectordbkwargs": vectordbkwargs
                 })
    answer = result["answer"]
    chat_history.append((query, answer))
    return answer, print_chat_history(chat_history)

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

    #Add This Conversation to the Memory Vectorstore
    loader = TextLoader(txt_file_path)
    document = loader.load()        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunked_conv = []
    texts = text_splitter.split_documents(document)
    chunked_conv.append(texts)
    for chunks in chunked_conv:
        Pinecone.from_texts([chunk.page_content for chunk in chunks], embeddings, index_name=index_name) 

    return f"Chat history saved and added to the VectorStore"
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


#Steamlit Test
#import streamlit as st

# #Streamlit layout and components
# st.set_page_config(page_title="Pinecone Q&A", layout="wide")
# st.title("Pinecone Q&A")

# user_input = st.text_input("Your Question:")
# submit_button = st.button("Ask Question")

# if submit_button:
#     answer, updated_chat_history = ask_question(user_input)
#     st.write(f"Current Answer: {answer}")
#     with st.beta_expander("History so far:"):
#         for q, a in updated_chat_history:
#             st.write(f"Q: {q}\nA: {a}\n")

# save_button = st.button("Save Conversation")
# if save_button:
#     save_status = save_chat_history()
#     st.write(f"Status: {save_status}")
