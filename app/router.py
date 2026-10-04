"""Router module per SPEC.md section 6 and Hard Rule 9.

Routes user requests to:
- "rag" (PDFs / document retrieval)
- "github" (GitHub repo / issue / PR / code queries)
- "calendar" (events, availability, scheduling)
- "email" (drafts, email search, email sending)
- "workflow" (multi-step operations combining calendar and email)
- "chitchat" (general conversation / greetings)

The router ONLY routes and NEVER calls tools (Hard Rule 9).
"""
import re
from typing import Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.config import settings
from app.logging import get_logger
from app.state import RouteDecision, Intent

logger = get_logger("router")

ROUTER_SYSTEM_PROMPT = """You are the intelligent intent router for a personal multi-agent assistant.
Classify the user's message into exactly ONE of the following intents:
- 'rag': Questions about ingested PDF documents, educational/technical concepts (such as data structures, trees, algorithms, handbooks, policies, definitions, or questions asking to explain, summarize, or retrieve factual knowledge).
- 'github': Queries about GitHub repositories, issues, pull requests, commits, or code.
- 'calendar': Checking calendar availability, finding free slots, creating, updating, or deleting calendar events, or asking questions about scheduled meetings.
- 'email': Drafting emails, reading emails, searching inbox, sending emails, or scheduling email dispatch.
- 'workflow': Multi-step requests combining calendar and email (e.g. 'schedule a meeting and email them', 'book a slot then email the team'). When selecting workflow, set workflow='schedule_and_email'.
- 'chitchat': Pure conversational greetings (e.g. 'hi', 'hello', 'who are you') without any request for documents, tasks, or tools.

CRITICAL ROUTING RULES:
1. For questions about concepts, definitions, educational topics, data structures (e.g. 'types of trees in data structures', 'what are trees', 'explain binary search tree'), or PDF content, route to 'rag'.
2. Consider the workspace context if provided. A user in the Documents workspace asking questions is seeking knowledge from documents ('rag'). A user in GitHub asking about repositories or issues is seeking GitHub operations.
3. For multi-turn follow-ups (e.g. 'explain the second one', 'show issues for the first one', 'which one is at 4pm'), preserve the ongoing conversation's intent.
Output a JSON object matching RouteDecision."""


def rule_based_route(
    query: str,
    history: list[Any] | None = None,
    workspace: str | None = None,
    agent_type: str | None = None,
) -> RouteDecision:
    """Heuristic rule-based fallback routing with workspace and history awareness."""
    q_lower = query.strip().lower()
    ws = (workspace or "").strip().lower()
    at = (agent_type or "").strip().lower()

    # 1. Multi-step workflow detection (combines calendar and email)
    has_calendar = any(w in q_lower for w in ["meeting", "schedule a meeting", "book a slot", "calendar", "event"])
    has_email = any(w in q_lower for w in ["and email", "then email", "and send email", "and mail", "email them"])
    if (has_calendar and has_email) or ("schedule_and_email" in q_lower):
        return RouteDecision(
            intent="workflow",
            workflow="schedule_and_email",
            reasoning="User requested multi-step action combining calendar scheduling and email notification.",
        )

    # 2. History-aware follow-up resolution (e.g., "send it", "confirm", "proceed", "explain the second one")
    confirmation_patterns = [
        r"^(yes|yep|yeah|sure|confirm|proceed|go ahead|send it|yes send|send that|send|correct)$"
    ]
    is_confirmation = any(re.match(p, q_lower) for p in confirmation_patterns)
    is_referential_follow_up = any(w in q_lower for w in [
        "the second one", "the first one", "the third one", "second one", "first one", "third one",
        "which one", "tell me more", "explain it", "more details", "what about"
    ])

    if (is_confirmation or is_referential_follow_up) and history:
        # Scan history backwards for context
        for msg in reversed(history):
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                c_lower = content.lower()
                # Check calendar context
                if any(w in c_lower for w in ["event", "calendar", "meeting", "slot", "do you mean", "tomorrow's schedule"]):
                    return RouteDecision(
                        intent="calendar",
                        reasoning="Follow-up referring to calendar event or scheduling context in history.",
                    )
                # Check email context
                if any(w in c_lower for w in ["draft", "email", "recipient", "subject", "message"]):
                    return RouteDecision(
                        intent="email",
                        reasoning="Follow-up referring to email drafting or sending in history.",
                    )
                # Check github context
                if any(w in c_lower for w in ["repository", "repositories", "pull request", "issue", "commit", "github"]):
                    return RouteDecision(
                        intent="github",
                        reasoning="Follow-up referring to GitHub repositories or issues in history.",
                    )
                # Check RAG context
                if any(w in c_lower for w in ["tree", "trees", "document", "pdf", "page", "citation", "passage", "data structure"]):
                    return RouteDecision(
                        intent="rag",
                        reasoning="Follow-up referring to RAG document retrieval in history.",
                    )

    # 3. Workspace contextual bias (if inside dedicated agent workspace)
    if ws in ("documents", "rag") or at == "rag":
        # Strongly bias towards RAG unless clearly invoking another domain
        if not any(w in q_lower for w in ["github", "pull request", "pr ", "commit", "calendar", "meeting", "schedule a", "email"]):
            return RouteDecision(intent="rag", reasoning="Query issued in Documents workspace targeting knowledge.")

    if ws == "github" or at == "github":
        if not any(w in q_lower for w in ["calendar", "meeting", "schedule", "email", "mail", "pdf", "document"]):
            return RouteDecision(intent="github", reasoning="Query issued in GitHub workspace targeting repositories/issues.")

    if ws == "calendar" or at == "calendar":
        if not any(w in q_lower for w in ["github", "pull request", "commit", "email", "draft", "pdf", "document"]):
            return RouteDecision(intent="calendar", reasoning="Query issued in Calendar workspace targeting schedule.")

    if ws == "email" or at == "email":
        if not any(w in q_lower for w in ["github", "repository", "calendar", "meeting", "pdf", "document"]):
            return RouteDecision(intent="email", reasoning="Query issued in Email workspace targeting email operations.")

    # 4. Explicit GitHub keywords
    if any(w in q_lower for w in ["github", "repo", "repository", "pull request", "pr ", "commit", "issue", "issues"]):
        return RouteDecision(intent="github", reasoning="Query targets GitHub repository or issue information.")

    # 5. Explicit Calendar keywords
    if any(w in q_lower for w in ["calendar", "meeting", "event", "events", "free slots", "slot", "slots", "conflicts", "book a", "schedule"]):
        return RouteDecision(intent="calendar", reasoning="Query targets calendar events or availability.")

    # 6. Explicit Email keywords
    if any(w in q_lower for w in ["email", "mail", "draft", "inbox", "send to", "write to", "write an email", "compose"]):
        return RouteDecision(intent="email", reasoning="Query targets email drafting, sending, or reading.")

    # 7. Semantic RAG & Technical Concept keywords (e.g. Data Structures, Trees, Algorithms, PDFs, Handbooks)
    rag_patterns = [
        "pdf", "document", "documents", "handbook", "policy", "search documents", "ingested",
        "tree", "trees", "data structure", "data structures", "binary search", "binary tree",
        "avl", "b-tree", "heap", "traversal", "nodes", "algorithm", "algorithms",
        "explain", "what is a", "what are the", "types of", "according to", "summary of"
    ]
    if any(w in q_lower for w in rag_patterns):
        return RouteDecision(intent="rag", reasoning="Query targets technical concepts or document knowledge.")

    # 8. Greetings / Chitchat
    if any(q_lower.startswith(w) for w in ["hi", "hello", "hey", "who are you", "what can you do", "thanks", "thank you"]):
        return RouteDecision(intent="chitchat", reasoning="Conversational greeting or general query.")

    # Default fallback
    return RouteDecision(intent="chitchat", reasoning="General query fallback.")


def route(
    query: str,
    history: list[Any] | None = None,
    workspace: str | None = None,
    agent_type: str | None = None,
    llm: Any | None = None,
) -> RouteDecision:
    """Route user query to an intent, using LLM if available, else rules."""
    if llm is not None and not settings.MOCK_MODE:
        try:
            ws_context = f"\nActive Workspace: {workspace or 'supervisor'}\nSpecialist Agent: {agent_type or 'none'}"
            structured_llm = llm.with_structured_output(RouteDecision)
            messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT + ws_context)]
            if history:
                messages.extend(history[-4:])  # recent context
            messages.append(HumanMessage(content=query))
            decision = structured_llm.invoke(messages)
            if isinstance(decision, RouteDecision):
                logger.info("router_classified_via_llm", intent=decision.intent, workflow=decision.workflow, reasoning=decision.reasoning)
                return decision
        except Exception as exc:
            logger.warn("router_llm_failed_falling_back_to_rules", error=str(exc))

    decision = rule_based_route(query, history=history, workspace=workspace, agent_type=agent_type)
    logger.info("router_classified_via_rules", intent=decision.intent, workflow=decision.workflow, reasoning=decision.reasoning)
    return decision
