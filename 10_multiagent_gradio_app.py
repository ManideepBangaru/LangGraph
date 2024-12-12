import gradio as gr
import pandas as pd
import plotly.express as px
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

# Global variable to cache the dataset and its path
cached_df = None
cached_file_path = None

def initialize_tools(file, query):
    global cached_df, cached_file_path
    try:
        # Check if the file path has changed
        if cached_file_path != file.name:
            # Load the uploaded file into a DataFrame
            try:
                cached_df = pd.read_csv(file.name)
                cached_file_path = file.name
                print("dataframe loaded successfully")
            except Exception as e:
                return f"Error loading the dataset: {e}", None
        else:
            print("Using cached dataframe")
        
        # Initialize the LLM model
        llm_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        # Define the Pandas DataFrame Tool
        def pandas_agent_tool(input_query: str):
            llm = llm_model
            pandas_agent = create_pandas_dataframe_agent(
                llm, 
                df=cached_df,
                allow_dangerous_code=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                verbose=False)
            response = pandas_agent.invoke({
                "input": input_query,
                "agent_scratchpad": f"Human: {input_query}\nAI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.\n\nAction: python_repl_ast\nAction Input: ",
            })

            # Check if the response contains Python code for Plotly
            if "px." in response:
                exec_globals = {}
                try:
                    # Execute the Python code in a controlled environment
                    print("entered plot section ....")
                    exec(response, {"df": cached_df, "px": px}, exec_globals)
                    fig = exec_globals.get("fig")
                    print(fig)
                    if fig:
                        return None, fig  # Return the Plotly figure
                except Exception as e:
                    return f"Error executing Plotly code: {e}", None

            # If the response is not a Plotly figure
            return response, None

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

        # System message
        sys_msg = SystemMessage(content="You are a helpful assistant tasked with bringing insights and doing mathematical operations on datasets")

        # Node
        def assistant(state: MessagesState):
            return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

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

        messages = [HumanMessage(content=query)]
        response = react_graph.invoke({"messages": messages})
        return response["messages"][-1].content, None

    except Exception as e:
        return f"Error: {e}", None

# Gradio interface
def gradio_app(file, query):
    if file is None or query.strip() == "":
        return "Please upload a CSV file and provide a query.", None
    response, plot = initialize_tools(file, query)
    return response, plot

# Create Gradio components
with gr.Blocks() as app:
    gr.Markdown("# DataFrame Query App")
    gr.Markdown("Upload a CSV file, enter a query, and get a response based on the data in the file.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"], interactive=True)
            query_input = gr.Textbox(label="Enter Query", placeholder="e.g., What is the average of column X?", lines=2)
            query_button = gr.Button("Submit")
        
        with gr.Column():
            response_output = gr.Markdown(label="Response")
            plot_output = gr.Plot(label="Plot Output")  # Renders Plotly figures interactively

    query_button.click(gradio_app, inputs=[file_input, query_input], outputs=[response_output, plot_output])

# Launch the app
if __name__ == "__main__":
    app.launch()