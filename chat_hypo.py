from langchain_openai import OpenAI
from langchain.chains import LLMChain, APIChain
# from prompts import ice_cream_assistant_prompt, api_response_prompt, api_url_prompt
from langchain.memory.buffer import ConversationBufferMemory
# from api_docs import scoopsie_api_docs
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from math import sqrt
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core.tools import tool

import chainlit as cl
import os

load_dotenv()
chat_history = []
opposite = 0
adajacent = 0
hypotenuse = 0
hypotenuseDone = False
openai_api_key = os.getenv("OPEN_AI_KEY")
@tool("PythagorasTool", return_direct=False)
def PythagorasTool(adjacent_side: str = None, opposite_side: str = None) -> float:
  """
  use this tool when you need to calculate the length of an hypotenuse
  given two sides of a triangle.
  To use the tool you must provide both of the following parameters: adjacent_side, and opposite_side. provide them in separate input values not together in one string
  """
  global opposite, adajacent, hypotenuse, hypotenuseDone
  # check for the values we have been given
  print(adjacent_side)
  print(opposite_side)

  if adjacent_side and opposite_side:
    opposite = opposite_side
    adajacent = adjacent_side
    hypotenuse = sqrt(float(adjacent_side)**2 + float(opposite_side)**2)
    hypotenuseDone = True
    return hypotenuse
  else:
    return "Could not calculate the hypotenuse of the triangle. Need both `adjacent_side`, `opposite_side`."
  
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
  You are a math assistant at CMU that helps user calculate the hypothenus of a triangle given both its adjacent and opposite sides.
  Use the PythagorasTool to calculate the length of an hypotenuse given two sides of a triangle.

  Before proceeding to the next step, ensure the prior one is fully completed. Please don't mention any STEP ; just act on it.
  Be sure that you do not skip any steps.

  Step #0: For any questions not related to triangles just answer with your common knowledge

  Step #1: Determine what the lengths of both sides are
- If the user input does not specify either of the opposite or adjacent sides or neither of them, then ask the user for any missing values.  Do not fill these in yourself.
- Confirmation Check: tell the user what the lengths of both sides are.

  Step #2: Determine the length of the hypotenuse
- Use the PythagorasTool to calculate the length of an hypotenuse given two sides of a triangle.
- Confirmation Check: Inform the user of the length of the hypotenuse.

  Step #3: Complete the task

            '''
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)  

@cl.on_chat_start
def setup_chain():
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
    tools = [PythagorasTool]
    llm_with_tools = llm.bind_tools(tools)

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


@cl.on_message
async def handle_message(message: cl.Message):
    global opposite, adajacent, hypotenuse, hypotenuseDone

    user_message = message.content.lower()
    llm_chain = cl.user_session.get("llm_chain")

    result = llm_chain.invoke({"input": user_message, "chat_history": chat_history})
    chat_history.extend(
    [
        HumanMessage(content=user_message),
        AIMessage(content=result["output"]),
    ]
    )
    if hypotenuseDone == False:  # not yet done, keep going around
        await cl.Message(result['output']).send()
    else:
        # send the add request to the UI
          
        fn = cl.CopilotFunction(name="formfill", args={"fieldA": opposite, "fieldB": adajacent, "fieldC": hypotenuse})
        hypotenuseDone = False
        res = await fn.acall()
        await cl.Message(content="Form info sent").send()

    
