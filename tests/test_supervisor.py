"""Tests for intent routing and supervisor orchestration per SPEC.md section 19.

Covers:
- Each intent routes correctly (rag, github, calendar, email, chitchat)
- Multi-step requests route to workflow ('schedule_and_email')
- Follow-up queries ('send it') use conversation history to route correctly
- End-to-end execution of capabilities through ChatService
"""
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import pytest

from app.router import route, rule_based_route
from app.service import ChatService
from app.testing.fakes import FakeGmailClient, FakeCalendarClient, FakeGitHubClient
from app.mcp_servers import gmail_server, calendar_server, github_server


@pytest.fixture(autouse=True)
def inject_fake_clients():
    """Inject fakes for all external integrations."""
    fake_gmail = FakeGmailClient()
    fake_cal = FakeCalendarClient()
    fake_gh = FakeGitHubClient()

    gmail_server.set_gmail_client(fake_gmail)
    calendar_server.set_calendar_client(fake_cal)
    github_server.set_github_client(fake_gh)

    yield {
        "gmail": fake_gmail,
        "calendar": fake_cal,
        "github": fake_gh,
    }

    gmail_server.set_gmail_client(None)
    calendar_server.set_calendar_client(None)
    github_server.set_github_client(None)


def test_intent_routing_rag():
    """Queries targeting ingested PDF documentation route to 'rag'."""
    d1 = rule_based_route("What does the company vacation policy say about carryover?")
    assert d1.intent == "rag"

    d2 = rule_based_route("Search the handbook document for remote work guidelines.")
    assert d2.intent == "rag"


def test_intent_routing_github():
    """Queries targeting GitHub repositories, PRs, or issues route to 'github'."""
    d1 = rule_based_route("List open issues in octocat/Hello-World.")
    assert d1.intent == "github"

    d2 = rule_based_route("Show me the latest pull request on owner/repo.")
    assert d2.intent == "github"


def test_intent_routing_calendar():
    """Queries targeting calendar events or availability route to 'calendar'."""
    d1 = rule_based_route("What events do I have scheduled for tomorrow?")
    assert d1.intent == "calendar"

    d2 = rule_based_route("Find free slots between 9am and 5pm tomorrow.")
    assert d2.intent == "calendar"


def test_intent_routing_email():
    """Queries targeting email drafting or search route to 'email'."""
    d1 = rule_based_route("Please draft an email to bob@example.com about the release.")
    assert d1.intent == "email"

    d2 = rule_based_route("Search my inbox for messages from manager@example.com.")
    assert d2.intent == "email"


def test_intent_routing_workflow_multistep():
    """Queries combining calendar scheduling and emailing route to 'workflow'."""
    query = "Schedule a meeting with alice@example.com for tomorrow at 2pm and email her the agenda."
    decision = rule_based_route(query)

    assert decision.intent == "workflow"
    assert decision.workflow == "schedule_and_email"

    query2 = "Book a slot for Team Sync tomorrow then email the team."
    decision2 = rule_based_route(query2)
    assert decision2.intent == "workflow"


def test_follow_up_uses_conversation_history():
    """Follow-up phrases like 'send it' route based on recent message history."""
    # Context 1: Prior email draft discussion
    history_email = [
        HumanMessage(content="Can you draft an email to client@example.com?"),
        AIMessage(content="Draft created for client@example.com with subject Project Quote."),
    ]
    d_email = rule_based_route("send it", history=history_email)
    assert d_email.intent == "email"

    # Context 2: Prior calendar discussion
    history_cal = [
        HumanMessage(content="Is 3pm tomorrow open for a meeting?"),
        AIMessage(content="Yes, 3:00 PM is free. Shall I create the calendar event?"),
    ]
    d_cal = rule_based_route("confirm", history=history_cal)
    assert d_cal.intent == "calendar"


def test_chitchat_routing():
    """Conversational greetings route to 'chitchat'."""
    d = rule_based_route("Hello there! Who are you?")
    assert d.intent == "chitchat"


def test_end_to_end_service_routing():
    """ChatService executes full flow through router and responder with MemorySaver."""
    service = ChatService(checkpointer=MemorySaver())

    # 1. Chitchat greeting
    res1 = service.send("thread_e2e_1", "Hi, what can you do?")
    assert res1["status"] == "completed"
    assert "assistant" in res1["response"].lower() or "help" in res1["response"].lower()

    # 2. Email drafting
    res2 = service.send("thread_e2e_2", "Draft an email to partner@example.com about Partnership saying Hello")
    assert res2["status"] == "completed"
    assert res2["intent"] == "email"
    assert "draft" in res2["response"].lower()

    # 3. Calendar events
    res3 = service.send("thread_e2e_3", "What events do I have tomorrow?")
    assert res3["status"] == "completed"
    assert res3["intent"] == "calendar"

    # 4. Workflow routing (triggers approval in Phase 7)
    res4 = service.send("thread_e2e_4", "Schedule a meeting with sam@example.com and email him")
    assert res4["status"] == "awaiting_approval"
    assert res4["pending_action"]["step"] == "create_event"
    # Approve event
    res4_2 = service.resume("thread_e2e_4", {"approved": True})
    assert res4_2["status"] == "awaiting_approval"
    # Approve email
    res4_3 = service.resume("thread_e2e_4", {"approved": True})
    assert res4_3["status"] == "completed"
    assert res4_3["intent"] == "workflow"
