"""
tests/test_agent.py

Integration tests for the LangGraph ReAct agent (ARIA).
Verifies multi-tool use and anti-hallucination behavior.
Requires GROQ_API_KEY in .env.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)

GROQ_AVAILABLE = bool(os.environ.get("GROQ_API_KEY"))


@pytest.mark.skipif(not GROQ_AVAILABLE, reason="GROQ_API_KEY not set")
class TestAgentIntegration:
    def test_agent_returns_answer(self):
        """Agent should return a non-empty string for any query."""
        from agents.disaster_agent import run_agent
        result = run_agent("What is the current weather situation?", thread_id="test_1")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 20

    def test_agent_uses_multiple_tools_for_flood_query(self):
        """
        For a complex operational query, the agent must use at least 2 tools.
        This validates the Agentic (not just RAG) behavior.
        """
        from agents.disaster_agent import run_agent
        result = run_agent(
            "Should we issue a mandatory evacuation near Boulder Creek? "
            "Consider water levels, weather, and any official protocols.",
            thread_id="test_2"
        )
        assert len(result["tool_calls"]) >= 2, (
            f"Expected ≥2 tool calls for a complex flood query, got: {result['tool_calls']}"
        )

    def test_agent_refuses_to_hallucinate(self):
        """
        For a completely unknowable question, the agent must not invent an answer.
        It should acknowledge lack of data.
        """
        from agents.disaster_agent import run_agent
        result = run_agent(
            "How many people are currently sheltering in Building XYZ-789?",
            thread_id="test_3"
        )
        answer_lower = result["answer"].lower()
        # Any of these phrases indicates the agent correctly declined to hallucinate
        hallucination_guards = [
            "not available", "don't have", "no information",
            "cannot find", "unavailable", "not in", "no data"
        ]
        assert any(guard in answer_lower for guard in hallucination_guards), (
            f"Agent may have hallucinated: {result['answer'][:200]}"
        )

    def test_agent_tool_calls_are_strings(self):
        """Tool call names should be strings (tool name registry validation)."""
        from agents.disaster_agent import run_agent
        result = run_agent("What is the current river level?", thread_id="test_4")
        for tc in result["tool_calls"]:
            assert isinstance(tc, str)
