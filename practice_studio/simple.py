from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated, Sequence, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

from dotenv import load_dotenv
load_dotenv()

# Custom state that extends MessagesState with coordination info
import operator
from typing import Annotated

class CoordinationState(MessagesState):
    # Supervisor decisions
    query_type: str = ""  # "fitness", "general", "both"
    supervisor_instructions: str = ""
    
    # Coordinator tracking
    agents_to_execute: list = []
    execution_plan: str = ""
    
    # Agent responses - using reducer to collect responses from multiple agents
    agent_responses: Annotated[list, operator.add] = []
    
    # Final curation
    user_query: str = ""


SupervisorLLM = init_chat_model(model="gpt-4o-mini", temperature=0)
CoordinatorLLM = init_chat_model(model="gpt-4o-mini", temperature=0)
FitnessAssistantLLM = init_chat_model(model="gpt-4o-mini", temperature=0)
GeneralAssistantLLM = init_chat_model(model="gpt-4o-mini", temperature=0)
ResponseCuratorLLM = init_chat_model(model="gpt-4o-mini", temperature=0)

# System messages
sys_msg_supervisor = SystemMessage(content="""You are a supervisor agent that analyzes user queries and provides instructions to a coordinator agent.

Your job is to:
1. Analyze the user's query to understand what type of assistance is needed
2. Classify the query as: "fitness", "general", or "both"
3. Provide clear instructions to the coordinator agent about which specialist agents to use

Respond with a JSON object containing:
{
    "query_type": "fitness" | "general" | "both",
    "instructions": "Clear instructions for the coordinator about which agents to use and how to approach the query"
}

Query types:
- "fitness": Fitness, exercise, health, nutrition, workouts, training, diet, wellness topics
- "general": Greetings, general questions, non-fitness topics
- "both": Complex queries that benefit from multiple perspectives""")

sys_msg_fitness = SystemMessage(content="""You are a fitness assistant agent that MUST strictly follow these instructions:
- You are only allowed to answer questions related to Fitness world
- Respond to in general greetings and ask user to be specific about the queries which you are allowed to answer
- If a request doesn't match any instruction, politely decline and direct to user on what you are allowed to do
- Never improvise or make assumptions beyond your explicit instructions
- Stay focused only on your assigned tasks
""")

sys_msg_general = SystemMessage(content="""You are a helpful general assistant. 
    For fitness-related questions, please direct users to ask more specifically about fitness topics.
    For any other questions, politely decline and direct to user on what you are allowed to do.""")

sys_msg_coordinator = SystemMessage(content="""You are a coordinator agent that manages the execution of specialist agents based on supervisor instructions.

Your responsibilities:
1. Receive instructions from the supervisor about which agents to use
2. Create an execution plan for calling the appropriate specialist agents
3. Manage the flow of information between agents
4. Collect responses from specialist agents and prepare them for the response curator

Based on the supervisor's instructions, you should:
- For "fitness" queries: Use only the fitness assistant
- For "general" queries: Use only the general assistant  
- For "both" queries: Use both fitness and general assistants

Return a JSON object with:
{
    "execution_plan": "Description of your execution approach",
    "agents_to_execute": ["fitness", "general"] // list of agents to call
}""")

sys_msg_response_curator = SystemMessage(content="""You are a response curator agent that shapes the final response to the user.

Your job is to:
1. Take the raw responses from specialist agents
2. Consider the original user query
3. Create a well-formatted, coherent, and helpful final response

Guidelines:
- If multiple agents responded, integrate their responses naturally
- Maintain the expertise and accuracy of the original responses
- Format the response to be clear and user-friendly
- Address the user's original question directly
- If responses conflict, present both perspectives clearly""")

# Nodes
def SupervisorAgent(state: CoordinationState):
    """Supervisor agent that analyzes queries and provides instructions to coordinator"""
    # Get the user message and store it for later use
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # Use LLM to analyze the query and provide instructions
    response = SupervisorLLM.invoke([sys_msg_supervisor, HumanMessage(content=user_message)])
    
    try:
        import json
        supervision_result = json.loads(response.content)
        query_type = supervision_result.get("query_type", "general")
        instructions = supervision_result.get("instructions", "")
    except:
        # Fallback if JSON parsing fails
        query_type = "general"
        instructions = "Handle as a general query"
    
    return {
        "query_type": query_type,
        "supervisor_instructions": instructions,
        "user_query": user_message
    }

def CoordinatorAgent(state: CoordinationState):
    """Coordinator agent that creates execution plan based on supervisor instructions"""
    query_type = state.get("query_type", "general")
    supervisor_instructions = state.get("supervisor_instructions", "")
    
    # Create execution plan
    coordination_prompt = f"""
    Supervisor Instructions: {supervisor_instructions}
    Query Type: {query_type}
    
    Based on these instructions, create an execution plan for handling this query.
    """
    
    response = CoordinatorLLM.invoke([sys_msg_coordinator, HumanMessage(content=coordination_prompt)])
    
    try:
        import json
        coordination_result = json.loads(response.content)
        execution_plan = coordination_result.get("execution_plan", "")
        agents_to_execute = coordination_result.get("agents_to_execute", [])
    except:
        # Fallback if JSON parsing fails
        if query_type == "fitness":
            agents_to_execute = ["fitness"]
        elif query_type == "general":
            agents_to_execute = ["general"]
        else:
            agents_to_execute = ["fitness", "general"]
        execution_plan = f"Execute {', '.join(agents_to_execute)} agent(s)"
    
    return {
        "execution_plan": execution_plan,
        "agents_to_execute": agents_to_execute
    }



def FitnessAssistant(state: CoordinationState):
    """Fitness specialist agent"""
    if "fitness" in state.get("agents_to_execute", []):
        response = FitnessAssistantLLM.invoke([sys_msg_fitness] + state["messages"])
        agent_response = {
            "agent_name": "fitness_assistant",
            "content": response.content,
            "timestamp": "fitness_completed"
        }
        return {"agent_responses": [agent_response]}
    return {"agent_responses": []}

def GeneralAssistant(state: CoordinationState):
    """General specialist agent"""
    if "general" in state.get("agents_to_execute", []):
        response = GeneralAssistantLLM.invoke([sys_msg_general] + state["messages"])
        agent_response = {
            "agent_name": "general_assistant", 
            "content": response.content,
            "timestamp": "general_completed"
        }
        return {"agent_responses": [agent_response]}
    return {"agent_responses": []}



def ResponseCurator(state: CoordinationState):
    """Response curator that waits for all agent responses and creates beautiful structured output"""
    user_query = state.get("user_query", "")
    agent_responses = state.get("agent_responses", [])
    agents_to_execute = state.get("agents_to_execute", [])
    
    # Check if we have received responses from all expected agents
    received_agents = {resp["agent_name"] for resp in agent_responses}
    expected_agents = set()
    if "fitness" in agents_to_execute:
        expected_agents.add("fitness_assistant")
    if "general" in agents_to_execute:
        expected_agents.add("general_assistant")
    
    # Only proceed if we have all expected responses
    if not expected_agents.issubset(received_agents):
        # Not all responses received yet, return empty to wait
        return {}
    
    # Format responses for the curator
    formatted_responses = ""
    for response in agent_responses:
        formatted_responses += f"""
**{response['agent_name'].replace('_', ' ').title()}:**
{response['content']}

"""
    
    if not formatted_responses:
        formatted_responses = "No specialist responses available."
    
    curation_prompt = f"""
    Original User Query: {user_query}
    
    Specialist Agent Responses:
    {formatted_responses}
    
    Please create a comprehensive, well-structured, and beautifully formatted final response that:
    1. Directly addresses the user's query
    2. Integrates insights from all specialist agents naturally
    3. Uses proper formatting with headers, bullet points, or sections as appropriate
    4. Provides a cohesive and professional response
    5. If multiple perspectives are available, synthesize them thoughtfully
    """
    
    final_response = ResponseCuratorLLM.invoke([sys_msg_response_curator, HumanMessage(content=curation_prompt)])
    
    return {"messages": [final_response]}



# Graph with efficient parallel coordination architecture
builder = StateGraph(CoordinationState)

# Add nodes
builder.add_node("supervisor", SupervisorAgent)
builder.add_node("coordinator", CoordinatorAgent)
builder.add_node("fitness_assistant", FitnessAssistant)
builder.add_node("general_assistant", GeneralAssistant)
builder.add_node("response_curator", ResponseCurator)

# Build the flow: supervisor → coordinator → agents (parallel) → response_curator
builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "coordinator")

# Implement fan-out pattern: coordinator triggers both agents simultaneously
# Both agents will execute in parallel when needed, and response_curator waits for all
builder.add_edge("coordinator", "fitness_assistant")
builder.add_edge("coordinator", "general_assistant")

# Fan-in: Both agents feed into response_curator 
# The response_curator will wait for all expected responses using the reducer pattern
builder.add_edge("fitness_assistant", "response_curator")
builder.add_edge("general_assistant", "response_curator")

# Final step
builder.add_edge("response_curator", END)

graph = builder.compile()

# # Testing the new coordination architecture ------------------------------------------------------------

# # Test 1 - General query (should route to general assistant only)
# res1 = graph.invoke({"messages": [HumanMessage(content="Hello, My name is Manideep")]})
# print("=== Test 1: General Query ===")
# for m in res1["messages"]:
#     m.pretty_print()

# # Test 2 - Fitness query (should route to fitness assistant only)  
# res2 = graph.invoke({"messages": [HumanMessage(content="What is a good workout for beginners?")]})
# print("=== Test 2: Fitness Query ===")
# for m in res2["messages"]:
#     m.pretty_print()

# # Test 3 - Complex query requiring both agents
# res3 = graph.invoke({"messages": [HumanMessage(content="I need help with creating a fitness routine and also some general life advice about time management")]})
# print("=== Test 3: Both Agents Query ===")
# for m in res3["messages"]:
#     m.pretty_print()

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