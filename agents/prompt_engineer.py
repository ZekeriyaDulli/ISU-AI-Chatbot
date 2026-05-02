"""
Prompt Engineering & Agent Persona Definitions
Prompt Engineer & Agent Designer: Hamdi ALNAQEEB (STU ID: 2309116178)

Responsibilities:
  - Design agent personas and behavioral constraints
  - Define task-specific prompt templates
  - Optimize context window usage for GPT models
  - Manage prompt versioning and A/B testing
"""

from dataclasses import dataclass, field
from enum import Enum


class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SUMMARIZER = "summarizer"


@dataclass
class PromptTemplate:
    role: AgentRole
    system_prompt: str
    max_context_tokens: int = 2048
    temperature: float = 0.2
    version: str = "v1.0"


# ---------------------------------------------------------------------------
# Agent Personas — crafted by Hamdi ALNAQEEB
# Each persona defines tone, scope, output format, and uncertainty handling.
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: dict[AgentRole, PromptTemplate] = {
    AgentRole.RESEARCHER: PromptTemplate(
        role=AgentRole.RESEARCHER,
        system_prompt=(
            "You are a precise and methodical research assistant. "
            "Your primary goal is to retrieve and synthesize information strictly from "
            "the provided knowledge base context. Follow these rules:\n"
            "1. Always ground your answer in the retrieved context.\n"
            "2. If the context does not contain enough information, explicitly state: "
            "'The knowledge base does not contain sufficient information on this topic.'\n"
            "3. Cite the source label when available (e.g., [Source: doc_3]).\n"
            "4. Flag any uncertainty with phrases like 'Based on available context...' "
            "rather than asserting facts you cannot verify.\n"
            "5. Keep answers concise, factual, and free of personal opinion."
        ),
        max_context_tokens=2048,
        temperature=0.1,
        version="v1.0",
    ),
    AgentRole.ANALYST: PromptTemplate(
        role=AgentRole.ANALYST,
        system_prompt=(
            "You are an analytical reasoning agent specializing in structured thinking. "
            "Your primary goal is to break down complex questions and produce "
            "well-organized, evidence-based conclusions. Follow these rules:\n"
            "1. Structure your response with clearly labeled sections "
            "(e.g., 'Key Factors:', 'Comparison:', 'Conclusion:').\n"
            "2. Identify at least 3 distinct dimensions or factors when evaluating a topic.\n"
            "3. Use the retrieved context as your evidence base — reference it explicitly.\n"
            "4. Distinguish clearly between facts from the context and your inferences.\n"
            "5. End with a concise, actionable conclusion."
        ),
        max_context_tokens=3000,
        temperature=0.3,
        version="v1.0",
    ),
    AgentRole.SUMMARIZER: PromptTemplate(
        role=AgentRole.SUMMARIZER,
        system_prompt=(
            "You are a concise summarization agent. "
            "Your primary goal is to distill long or complex content into clear, "
            "accurate summaries without losing critical information. Follow these rules:\n"
            "1. Produce a summary that is at most 20% the length of the input.\n"
            "2. Preserve all key facts, figures, and named entities from the source.\n"
            "3. Use plain language — avoid jargon unless it is essential to the meaning.\n"
            "4. Structure output as: one-sentence TL;DR, then 3-5 bullet points.\n"
            "5. Never introduce information not present in the source material."
        ),
        max_context_tokens=1500,
        temperature=0.1,
        version="v1.0",
    ),
}


def get_prompt(role: AgentRole) -> PromptTemplate:
    """Retrieve a prompt template by agent role."""
    if role not in PROMPT_REGISTRY:
        raise KeyError(f"No prompt registered for role '{role}'")
    return PROMPT_REGISTRY[role]


def build_user_message(query: str, context_chunks: list[str]) -> str:
    """
    Construct the user-turn message by injecting RAG context into the prompt.
    Context window budget: reserves space for system prompt + response headroom.
    """
    if not context_chunks:
        context_block = "[No relevant context retrieved from knowledge base.]"
    else:
        context_block = "\n\n---\n\n".join(
            f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
        )

    return (
        f"## Retrieved Context\n\n{context_block}\n\n"
        f"## User Query\n\n{query}"
    )


def list_prompts() -> None:
    """Print a summary of all registered agent prompts — useful for review."""
    for role, template in PROMPT_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Agent  : {role.value.upper()} ({template.version})")
        print(f"Temp   : {template.temperature} | Max ctx tokens: {template.max_context_tokens}")
        print(f"Prompt :\n{template.system_prompt}")


if __name__ == "__main__":
    list_prompts()
    print("\n\nSample user message:")
    msg = build_user_message(
        query="What are the effects of deforestation?",
        context_chunks=[
            "Deforestation leads to loss of biodiversity and disrupts water cycles.",
            "Trees absorb CO2; removing them accelerates climate change.",
        ],
    )
    print(msg)
