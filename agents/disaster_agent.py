"""
agents/disaster_agent.py

The LangGraph ReAct Agentic Core.
Uses a Reason + Act loop to autonomously call tools, observe results, and deliver
grounded, cited, hallucination-free disaster situational reports.

Architecture:
  ChatGroq (Llama-3.3-70B) + 6 tools → LangGraph create_react_agent → MemorySaver
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import ALL_TOOLS

SYSTEM_PROMPT = """
You are ARIA — Agentic Response Intelligence for Disasters.

You are an expert AI analyst embedded with an emergency response team during an active disaster.
You have access to live data tools and a searchable knowledge base. Your job is to provide 
accurate, grounded, actionable situational assessments.

## Your Tools
- `fetch_live_weather`: Get real-time temperature, rainfall, wind from Open-Meteo
- `query_water_sensor`: Get real-time USGS river gage height and flood status
- `query_recent_earthquakes`: Get recent seismic events from USGS
- `search_knowledge_base`: Search FEMA/CDC protocols, historical tweets, satellite imagery descriptions
- `analyze_disaster_image`: Analyze a real satellite/drone image with local Vision AI
- `generate_structured_alert`: Log a formal severity-tagged emergency alert

## Critical Rules
1. ALWAYS use tools to gather real data before answering operational questions
2. CITE your sources: mention which tool or document you used and its timestamp
3. If you have conflicting data, acknowledge the conflict and cite the most recent source
4. NEVER hallucinate. If information is unavailable, say so explicitly
5. For multi-part questions (e.g. "should we evacuate?"), call at least 2 tools before concluding
6. After identifying a CRITICAL situation, always call generate_structured_alert
7. Keep operational answers concise and action-oriented. Responders are under time pressure.

## Output Format for Operational Queries
- **Status**: [NORMAL / WARNING / DANGER / CRITICAL]
- **Evidence**: [cite each data point and its source]
- **Recommendation**: [clear action items]
- **Alert Generated**: [yes/no — severity level]
"""

def get_agent():
    """
    Instantiates and returns the LangGraph ReAct disaster response agent.
    Includes conversation memory for multi-turn sessions.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT
    )
    return agent


def run_agent(query: str, thread_id: str = "default") -> dict:
    """
    Runs the agent with a query and returns the final answer + tool call trace.

    Args:
        query: The user's question or command
        thread_id: Conversation thread for multi-turn memory

    Returns:
        dict with keys: 'answer' (str), 'tool_calls' (list of str)
    """
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    tool_calls = []
    final_answer = ""

    try:
        for chunk in agent.stream(
            {"messages": [("human", query)]},
            config=config,
            stream_mode="values"
        ):
            messages = chunk.get("messages", [])
            for msg in messages:
                # Capture tool call names for the UI trace
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(tc.get("name", "unknown_tool"))
                # Capture the final AI response
                elif hasattr(msg, "content") and msg.content and msg.type == "ai":
                    final_answer = msg.content

    except Exception as e:
        final_answer = f"Agent error: {e}"

    return {
        "answer": final_answer,
        "tool_calls": list(dict.fromkeys(tool_calls))  # deduplicated, ordered
    }


if __name__ == "__main__":
    print("ARIA — Agentic Response Intelligence for Disasters")
    print("Type 'quit' to exit\n")
    thread = "cli_session"
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        result = run_agent(q, thread_id=thread)
        print(f"\nARIA: {result['answer']}")
        if result["tool_calls"]:
            print(f"  [Tools used: {', '.join(result['tool_calls'])}]\n")
