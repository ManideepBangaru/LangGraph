from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.tools import Tool
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
import pandas as pd

# Step 1: Create a DataFrame
df = pd.read_csv("movies_meta.csv")
df.head()

# initialize llm model
llm_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Step 2: Define the Pandas DataFrame Tool
def pandas_agent_tool(input_query: str) -> str:
    llm = llm_model
    pandas_agent = create_pandas_dataframe_agent(
        llm, 
        df=df,
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False)
    response = pandas_agent.invoke({
        "input": input_query,
        "agent_scratchpad": f"Human: {input_query}\nAI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.\n\nAction: python_repl_ast\nAction Input: ",
    })
    return response

pandas_tool = Tool(
    name="PandasAgentTool",
    func=pandas_agent_tool,
    description="Useful for answering questions about a pandas DataFrame."
)

# Step 3: Define the Multiplication Tool
def multiply_values(input_query: str) -> str:
    try:
        # Parse the input (e.g., "5 * 3")
        values = [float(x.strip()) for x in input_query.split("*")]
        result = values[0] * values[1]
        return f"The result of multiplication is {result}."
    except Exception as e:
        return f"Error in multiplication: {e}"

multiplication_tool = Tool(
    name="MultiplicationTool",
    func=multiply_values,
    description="Useful for performing multiplication of two values. Input format: 'a * b'."
)

# Step 4: Create the LangGraph
tools = [pandas_tool, multiplication_tool]
llm_with_tools = llm_model.bind_tools(tools)

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Show
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

if __name__ == "__main__":
    queryy = input("Enter your query : ")
    messages = [HumanMessage(content=queryy)]
    messages = react_graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
