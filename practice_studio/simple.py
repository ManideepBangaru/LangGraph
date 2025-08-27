from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from typing import Annotated, Sequence, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

from dotenv import load_dotenv
load_dotenv()

# Custom state that extends MessagesState with routing info
class RoutingState(MessagesState):
    route: str = ""


SupervisorLLM = init_chat_model(model="gpt-4o-mini", temperature=0)
FitnessAssistantLLM = init_chat_model(model="gpt-4o-mini", temperature=0)

# System message for routing supervisor
sys_msg_supervisor = SystemMessage(content="""You are a routing supervisor that analyzes user queries and determines where to direct them.
Your ONLY job is to classify the user's query into one of these categories:
- "fitness" - if the query is about fitness, exercise, health, nutrition, workouts, training, diet, wellness, etc.
- "general" - if the query is a general greeting, unclear, or not fitness-related
You MUST respond with ONLY ONE WORD: either "fitness" or "general"
Do NOT provide any other response. Just the classification word.""")

sys_msg_fitness = SystemMessage(content="""You are a fitness assistant agent that MUST strictly follow these instructions:
- You are only allowed to answer questions related to Fitness world
- Respond to in general greetings and ask user to be specific about the queries which you are allowed to answer
- If a request doesn't match any instruction, politely decline and direct to user on what you are allowed to do
- Never improvise or make assumptions beyond your explicit instructions
- Stay focused only on your assigned tasks
""")

# Nodes
def Supervisor(state: RoutingState):
    # Get the last user message for routing
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # Use LLM to classify the query
    classification = SupervisorLLM.invoke([sys_msg_supervisor, HumanMessage(content=user_message)])
    
    # Store the routing decision in state
    return {"route": classification.content.strip().lower()}

def GeneralAssistant(state: RoutingState):
    general_msg = SystemMessage(content="""You are a helpful general assistant. 
    For fitness-related questions, please direct users to ask more specifically about fitness topics.
    For other questions, provide helpful general assistance.""")
    return {"messages": [SupervisorLLM.invoke([general_msg] + state["messages"])]}

def FitnessAssistant(state: RoutingState):
    return {"messages": [FitnessAssistantLLM.invoke([sys_msg_fitness] + state["messages"])]}

# Routing function
def route_query(state: RoutingState) -> Literal["fitness_assistant", "general_assistant"]:
    route = state.get("route", "general")
    if "fitness" in route:
        return "fitness_assistant"
    else:
        return "general_assistant"

# Graph
builder = StateGraph(RoutingState)

# Add nodes
builder.add_node("supervisor", Supervisor)
builder.add_node("fitness_assistant", FitnessAssistant)
builder.add_node("general_assistant", GeneralAssistant)

# Add edges
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_query,
    {
        "fitness_assistant": "fitness_assistant",
        "general_assistant": "general_assistant"
    }
)
builder.add_edge("fitness_assistant", END)
builder.add_edge("general_assistant", END)

graph = builder.compile()

# Testing without memory ------------------------------------------------------------

# Test 1
res1 = graph.invoke({"messages": [HumanMessage(content="Hello, My name is Manideep")]})
for m in res1["messages"]:
    m.pretty_print()

# Test 2
res2 = graph.invoke({"messages": [HumanMessage(content="Do you remember my name ?")]})
for m in res2["messages"]:
    m.pretty_print()

# # Testing with memory ------------------------------------------------------------
# from langgraph.checkpoint.memory import MemorySaver
# memory = MemorySaver()
# graph_with_memory = builder.compile(checkpointer=memory)

# # Specify a thread
# config = {"configurable": {"thread_id": "1"}}

# res3 = graph_with_memory.invoke({"messages": [HumanMessage(content="Hello, My name is Manideep")]}, config)
# for m in res3["messages"]:
#     m.pretty_print()

# res4 = graph_with_memory.invoke({"messages": [HumanMessage(content="Do you remember my name ?")]}, config)
# for m in res4["messages"]:
#     m.pretty_print()