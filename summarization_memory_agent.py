import os
import tiktoken
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState

# 1. Initialization
load_dotenv()
model = ChatOpenAI(model="gpt-4o", temperature=0.2) 

class State(MessagesState):
    summary: str

# 2. Operational Nodes
def call_model(state: State, config: RunnableConfig):
    summary = state.get("summary", "")
    if summary:
        system_instruction = (
            "You are a Senior Cloud Infrastructure Support Engineer. "
            f"Summary of technical troubleshooting performed so far: {summary}"
        )
    else:
        system_instruction = (
            "You are a Senior Cloud Infrastructure Support Engineer helping a developer "
            "troubleshoot production environment failures. Be concise, technical, and precise."
        )
    
    messages = [SystemMessage(content=system_instruction)] + state["messages"]
    response = model.invoke(messages, config)
    return {"messages": response}

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_prompt = (
            f"Current technical summary: {summary}\n\n"
            "Extend this summary with the newest diagnostic logs and messages above. "
            "Keep it dense with technical facts, error codes, and resource IDs."
        )
    else:
        summary_prompt = (
            "Synthesize the preceding troubleshooting conversation into a concise technical summary. "
            "Focus on the root cause, system specs, error messages, and actions taken."
        )

    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    response = model.invoke(messages)
    
    # Prune historical messages leaving only the last two active responses
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# 3. Production Token Router
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    messages = state["messages"]
    encoding = tiktoken.encoding_for_model("gpt-4o")
    
    # Calculate exact active token weight
    total_tokens = sum(len(encoding.encode(m.content)) for m in messages)
    
    # In production, use standard limits (e.g., 4000). For demonstration testing, we can use 300 tokens.
    TOKEN_THRESHOLD = 300 
    
    if total_tokens > TOKEN_THRESHOLD:
        return "summarize_conversation"
    return END

# 4. Graph Construction
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# 5. CLI Execution Loop
if __name__ == "__main__":
    # Generate visual asset locally in workspace
    try:
        with open("architecture_graph.png", "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        print("[System] Visual asset successfully compiled to 'architecture_graph.png'")
    except Exception as e:
        print(f"[System] Skip visual rendering: {e}")

    config = {"configurable": {"thread_id": "cloud-support-session-01"}}
    print("\n" + "="*60)
    print("  LANGGRAPH CLOUD SUPPORT AGENT DEMO ")
    print("="*60 + "\n🤖 Support Active. Enter infrastructure issues or logs:\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        hesitate = user_input.strip()
        if not hesitate:
            continue

        events = graph.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="updates")
        for event in events:
            if "conversation" in event:
                print(f"\n🤖 Assistant: {event['conversation']['messages'].content}\n")
            if "summarize_conversation" in event:
                print("="*60 + "\n⚡ [LANGGRAPH INTERNAL ACTION: CONTEXT COMPRESSION]")
                print(f"New Summary:\n{event['summarize_conversation']['summary']}")
                print(f"Evicted Messages: {len(event['summarize_conversation']['messages'])} items.")
                print("="*60 + "\n")
