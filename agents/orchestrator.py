"""
Multi-Agent Orchestrator
Lead Developer & System Architect: Zekeriya Dulli (STU ID: 2309115377)
Prompt Engineer & Agent Designer: Hamdi ALNAQEEB (STU ID: 2309116178)
"""

import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.safety_monitor import SafetyMonitor
from agents.prompt_engineer import AgentRole, get_prompt, build_user_message
from rag.vector_db import VectorDBClient

load_dotenv()


class Agent:
    """A single agent driven by a PromptTemplate from prompt_engineer.py."""

    def __init__(self, role: AgentRole, llm: ChatOpenAI, rag: VectorDBClient):
        self.name = role.value
        self._template = get_prompt(role)
        self.llm = llm
        self.rag = rag

    def run(self, user_query: str) -> str:
        context_docs = self.rag.query(user_query, n_results=4)
        user_message = build_user_message(user_query, context_docs)

        messages = [
            SystemMessage(content=self._template.system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self.llm.invoke(messages)
        return response.content


class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents, routes queries, and aggregates results.
    Integrates safety monitoring on every agent output before returning.
    """

    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
        base_url = os.getenv("LLM_BASE_URL")  # set to LM Studio URL to use a local model
        api_key = os.getenv("OPENAI_API_KEY", "lm-studio")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            base_url=base_url or None,
            api_key=api_key,
        )
        self.rag = VectorDBClient()
        self.safety = SafetyMonitor()

        self.agents: dict[str, Agent] = {
            role.value: Agent(role, self.llm, self.rag)
            for role in AgentRole
        }

    def _route(self, query: str) -> str:
        """Simple keyword router — extend with an LLM router for production."""
        q = query.lower()
        if any(w in q for w in ["summarize", "tldr", "brief", "shorten"]):
            return "summarizer"
        if any(w in q for w in ["analyze", "compare", "evaluate", "assess"]):
            return "analyst"
        return "researcher"

    REFUSAL = "I'm sorry, but I can't help with that. This topic has been flagged as harmful and will not be discussed."

    def run(self, user_query: str) -> dict[str, Any]:
        # Check the user's input BEFORE sending anything to the LLM
        input_safety = self.safety.check(user_query)
        if not input_safety["passed"]:
            return {
                "agent_used": "safety_monitor",
                "response": self.REFUSAL,
                "safety": input_safety,
                "blocked": True,
            }

        agent_name = self._route(user_query)
        agent = self.agents[agent_name]
        raw_output = agent.run(user_query)
        output_safety = self.safety.check(raw_output)

        # Also block if the LLM output itself is flagged
        if not output_safety["passed"]:
            return {
                "agent_used": agent_name,
                "response": self.REFUSAL,
                "safety": output_safety,
                "blocked": True,
            }

        return {
            "agent_used": agent_name,
            "response": raw_output,
            "safety": output_safety,
            "blocked": False,
        }


if __name__ == "__main__":
    orchestrator = MultiAgentOrchestrator()
    result = orchestrator.run("What are the main causes of climate change?")
    print(f"Agent: {result['agent_used']}")
    print(f"Response:\n{result['response']}")
    print(f"Safety: {result['safety']}")
