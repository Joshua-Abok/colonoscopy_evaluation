from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import sys




# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
)
from langchain.schema import SystemMessage    # helps make plain, simple system mess without any templating
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler



load_dotenv()

embeddings = OpenAIEmbeddings()

def process_file(file_path):
    # Your existing code here, replacing hard-coded file paths with the provided file_path
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=100, chunk_overlap=50)
    
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)
    
    empty_doc = []
    for doc in docs:
        store_docs = doc.page_content
        # print("our STORED DOCS", store_docs)
        empty_doc.append(store_docs)
    
    print('\n')
    full_text = ' '.join(empty_doc[1:])
    # print("\n\n\n THE FULL TEXT VALUES", full_text)
    
    colon_guidelines_loader = PyPDFLoader("colonoscopy-guidelines.pdf")
    colon_guidelines = colon_guidelines_loader.load()
    # print("COLON GUIDELINES", colon_guidelines[0].page_content)
    
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")
    
    results = db.similarity_search("CPT CODES", k=1)
    
    empty_lst = []
    for result in results:
        print("\n")
        content = result.page_content
        content_split = content.split(" ")[-1]
        print(f"the patient's CPT CODE: {content_split}")

    cpt_codes = content_split

    
    return full_text, colon_guidelines, cpt_codes

if __name__ == "__main__":
    # Check if a command-line argument (file path) is provided
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
    else:
        file_path = sys.argv[1]
        full_text, colon_guidelines, cpt_codes = process_file(file_path)





handler = ChatModelStartHandler()
chat = ChatOpenAI(
    model="gpt-4",
    callbacks=[handler]  
)




prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a colonoscopy guidelines. \n"
            f"The colonoscopy guidelines are as follows: {colon_guidelines}\n"
            "Do not make any assumptions about what doesn't exist in the guidelines"
            "or what guides exist. Instead, use only the provided colonoscopy guidelines"
            "and patient report to determine whether the criteria for medical necessity has been met."
            "Use and outline the evidence from medical record to support the decision"
            
            )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"), 
        MessagesPlaceholder(variable_name="agent_scratchpad")
            #      agent_scratchpad -> simplified form of memory (keeps track with convo with chatGPT)
    ]
)



# create a new memory object 
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# refactor -> variable for tools 
tools = [write_report_tool]

# chain that knows how to use tools 
# -> take list of tools & convert 'em into JSON function descr. 
# -> has input var, memory, prompts, etc - all stuff that a CHAIN has. 
agent = OpenAIFunctionsAgent(
    llm=chat, 
    prompt=prompt,
    tools=tools
)

# takes an agent and runs it until the response is not a function call -> while loop
agent_executor = AgentExecutor(
    agent=agent, 
    verbose=True,  #hv created handler now
    tools=tools,
    memory=memory
)


agent_executor(
    f"create a report evaluating the patient's record based on the colonoscopy guidelines: {full_text}\n"
    "again emphasis is on creating a report"
)

