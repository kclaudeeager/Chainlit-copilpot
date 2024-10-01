import asyncio
from langchain.agents import Tool
from typing import Union, Dict, Any
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
import chainlit as cl
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
import pickle
from concurrent.futures import ThreadPoolExecutor
load_dotenv()

# Initialize API keys
openai_api_key = os.getenv("OPEN_AI_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
retriever_tool = None
search = TavilySearchResults(api_key=tavily_api_key)
input_fields = []
# Global state for the conversation
conversation_state = {
    "setup_complete": False,
    "is_solution_ready": False,
    "app_name": None,
    "input_fields": [],
    "description": None,
    "code_snippet": None,
    "chat_history": [],
    "retriever_tool": None,
    "tools": []
}

CACHE_FILE_PATH = "faiss_index.index"
# Load PDF documents
def load_pdf_documents(directory):
    documents = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                pdf_loader = PyPDFLoader(file_path)
                pdf_document = pdf_loader.load()
                documents.extend(pdf_document)
            else:
                cl.Message(content=f"File path {file_path} is not a valid file.").send()
    return documents


# Create vectors from documents
def load_documents_turn_into_vectors(documents):
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(documents)
    vector = FAISS.from_documents(documents, embeddings)
    return vector



def save_vectors_to_cache(vector_store, cache_file=CACHE_FILE_PATH):
    try:
        # Save the FAISS index using FAISS built-in method
        vector_store.save_local(cache_file)
        print(f"FAISS index saved to cache file {cache_file}.")
    except Exception as e:
        print(f"Failed to save FAISS index to cache: {e}")


def load_vectors_from_cache(cache_file=CACHE_FILE_PATH):
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} does not exist. Regenerating vector.")
        return None

    try:
        # Load the FAISS index using FAISS built-in method
        vector_store = FAISS.load_local(cache_file, embeddings,allow_dangerous_deserialization=True)
        print("Loaded FAISS index from cache.")
        return vector_store
    except Exception as e:
        print(f"Failed to load FAISS index from cache: {e}. Regenerating vector.")
        return None
    
# Create a retrieval chain
def create_retrevial_chain(vector):
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "Programming_teacher",
        "Search for information about Rust and Kotlin programming languages and answer any asked questions. You must use this tool!"
    )
    conversation_state["retriever_tool"] = retriever_tool


# Define a tool for setting global variables
class SetGlobalVariablesTool(BaseTool):
    name: str = "set_global_variables"
    description: str = "Use this tool to set the values of conversation_state variables."

    async def _run(self, tool_input: Union[str, Dict[str, Any]]) -> str:
        return await self._arun(tool_input)

    async def _arun(self, tool_input: Union[str, Dict[str, Any]]) -> str:
        global conversation_state
        if isinstance(tool_input, dict):
            conversation_state.update(tool_input)
            return "Conversation state updated successfully."
        else:
            return "Invalid input format. Expected a dictionary."


# Define a tool for collecting input fields
class GetInputFieldsTool(BaseTool):
    name: str = "get_input_fields"
    description: str = "Collects app name and description from the user."

    async def _run(self, tool_input: Union[str, Dict[str, str]]) -> Dict[str, str]:
        return await self._arun(tool_input)

    async def _arun(self, tool_input: Union[str, Dict[str, str]]) -> Dict[str, str]:
        global input_fields, conversation_state
        if isinstance(tool_input, dict):
            prompts = list(tool_input.values())
        else:
            return {"error": "Invalid input format. Expected a dictionary."}

        collected_fields: Dict[str, str] = {}
        for prompt in prompts:
            field = await self.show_input_popup(prompt)
            collected_fields[prompt] = field
        input_fields = list(collected_fields.values())
        conversation_state["input_fields"] = input_fields
        return collected_fields

    async def show_input_popup(self, prompt: str) -> str:
        input_field = cl.Input(label=prompt, placeholder="Type your response here...")
        popup = cl.Popup(title="Input Required", content=input_field)
        await popup.send()
        response = await input_field.get_value()
        return response


# Define a tool for generating the app
class SimpleAppTool(BaseTool):
    name: str = "simple_app"
    description: str = "Stores and returns the simple app code generated by the agent."

    async def _arun(self, tool_input: Union[str, Dict[str, Any]]) -> str:
        print("Running simple app tool")
        print("Tool input: ", tool_input)
        global conversation_state
        # check the type of the input
        if isinstance(tool_input, dict):
            
            if tool_input.get("code_snippet") is None:
                return "Invalid input format. Expected a dictionary with 'tool_input' key."
            conversation_state["app_name"] = tool_input.get("app_name")
            conversation_state["description"] = tool_input.get("description")
            conversation_state["code_snippet"] = tool_input.get("code_snippet")
      
        conversation_state["is_solution_ready"] = True
    
        fn = cl.CopilotFunction(name="formfill", args={"fieldA": conversation_state['app_name'], "fieldB": conversation_state["description"], "fieldC": conversation_state["code_snippet"] })
        conversation_state["is_solution_ready"] = False
        res = await fn.acall()
        await cl.Message(content="Form info sent").send()
        
        return f"Here is the Python code snippet:\n\n{conversation_state}"

    def _run(self, app_name: str, description: str, code_snippet: str) -> str:
        # Synchronous version if needed
        tool_input = {"app_name": app_name, "description": description, "code_snippet": code_snippet}
        return asyncio.run(self._arun(tool_input))


# Define the conversational prompt
improved_prompt = '''
You are an expert assistant helping users create simple applications step by step. Follow these instructions carefully:

1. When the user mentions they want to create a simple app, guide them through each step of the process.
2. First, ask for the app name by saying: "Please provide the name of the application you want to create."
3. After the user provides the app name, confirm the name by saying: "Thank you. The app name is [app name]. Now, please describe the purpose or functionality of the app."
4. Once the user provides the description, confirm it with: "Got it. The description is [description]."
5. Remember to store the app name and description in the conversation state as `conversation_state["app_name"]` and `conversation_state["description"]`.
5. Now that you have both the app name and description, generate a simple app code snippet and pass it to `simple_app` tool by invoking it by passing the app name as `app_name`, description of what app do as `description` and generated codes as code_snippet
6. After the code snippet is generated, confirm the completion of the task and wait for the user to request the next step.
7. If any input is missing, prompt the user for the missing information and confirm each input before proceeding to the next step.

Make sure all steps are clearly communicated and proceed only after receiving the necessary information.
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", improved_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

@cl.on_chat_start
async def setup_chain():
    loading_message = await cl.Message(content="Setting up the system, please wait...").send()

    # Try to load the vector from cache
    vector = load_vectors_from_cache(CACHE_FILE_PATH)

    if vector is None:
        print("No cache found. Loading documents and creating vectors.")
        documents = load_pdf_documents("docs")
        vector = load_documents_turn_into_vectors(documents)
        save_vectors_to_cache(vector)

    create_retrevial_chain(vector)

    # Initialize tools
    get_input_fields_tool = GetInputFieldsTool()
    set_global_variables_tool = SetGlobalVariablesTool()
    simple_app_tool = SimpleAppTool()

    # Initialize the model
    llm = ChatOpenAI(
        openai_api_key=openai_api_key, model="gpt-3.5-turbo", streaming=True, max_tokens=1000
    )

    # Bind tools to the model
    tools = [conversation_state["retriever_tool"], get_input_fields_tool, set_global_variables_tool, simple_app_tool, search]
    llm_with_tools = llm.bind_tools(tools)

    # Create agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    cl.user_session.set("llm_chain", agent_executor)
    conversation_state["setup_complete"] = True
    loading_message.content = "System setup complete. You can now start interacting."
    await loading_message.update()

@cl.on_message
async def handle_message(message: cl.Message):
    global conversation_state
    setup_complete = conversation_state["setup_complete"]
    is_solution_ready = conversation_state["is_solution_ready"]
    input_fields = conversation_state["input_fields"]
    code_snippet = conversation_state["code_snippet"]
    if not setup_complete:
        await cl.Message(content="The system is still initializing. Please wait a moment and try again.").send()
        return

    user_message = message.content.lower()
    llm_chain = cl.user_session.get("llm_chain")

    result =  llm_chain.invoke({"input": user_message, "chat_history": conversation_state["chat_history"]})
    conversation_state["chat_history"].extend(
        [
            HumanMessage(content=user_message),
            AIMessage(content=result["output"]),
        ]
    )
    
    print("Is solution ready: ", is_solution_ready)
    if is_solution_ready:  # not yet done, keep going around
        fn = cl.CopilotFunction(name="formfill", args={"fieldA": input_fields[0], "fieldB": input_fields[1], "fieldC": code_snippet})
        conversation_state["is_solution_ready"] = False
        res = await fn.acall()
        await cl.Message(content="Form info sent").send()
       
    else:
        await cl.Message(result['output']).send()
    