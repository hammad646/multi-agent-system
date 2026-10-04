"""Tests for Phase 3: GitHub FastMCP Server, Agent, Error Taxonomy, and Guardrails."""
import io
import sys
import pytest

from app.config import settings
from app.mcp_servers import github_server
from app.mcp_servers.github_server import (
    search_repositories,
    get_repository,
    list_issues,
    get_issue,
    list_pull_requests,
    get_pull_request,
    list_commits,
    get_file_contents,
    search_code,
    list_branches,
    get_user,
    create_issue,
    create_issue_comment,
    set_github_client,
)
from app.agents.github_agent import GitHubAgent
from app.testing.fakes import FakeGitHubClient


@pytest.fixture(autouse=True)
def setup_github_mock():
    """Ensure FakeGitHubClient is injected and restore state after test."""
    fake_client = FakeGitHubClient(token="mock-pat")
    set_github_client(fake_client)
    prev_write = settings.GITHUB_ALLOW_WRITE
    settings.GITHUB_ALLOW_WRITE = False
    yield fake_client
    settings.GITHUB_ALLOW_WRITE = prev_write
    set_github_client(None)


# --- 1. Acceptance Checks ---

def test_show_readme_works():
    """Verify 'show README' retrieves file content wrapped in <data> tags."""
    agent = GitHubAgent()
    res = agent.invoke("Show README in octocat/Hello-World")

    assert res.status == "ok"
    assert "README for octocat/Hello-World" in res.summary
    assert "# Hello-World" in res.summary
    assert "<data>" in res.summary
    assert "</data>" in res.summary
    assert res.data["file"] == "README.md"


def test_open_issues_in_repo_works():
    """Verify 'open issues in <repo>' retrieves issues and includes full URLs."""
    agent = GitHubAgent()
    res = agent.invoke("List open issues in octocat/Hello-World")

    assert res.status == "ok"
    assert "Open issues in octocat/Hello-World" in res.summary
    assert "#1 Bug in greeting message" in res.summary
    assert "https://github.com/octocat/Hello-World/issues/1" in res.summary
    assert len(res.data["issues"]) >= 2


def test_write_tools_disabled_by_default():
    """Verify write tools return permission_error when GITHUB_ALLOW_WRITE=false."""
    settings.GITHUB_ALLOW_WRITE = False

    # 1. create_issue
    res_issue = create_issue("octocat/Hello-World", "Test Issue", "Issue body")
    assert res_issue["ok"] is False
    assert res_issue["error"]["code"] == "permission_error"
    assert res_issue["error"]["retryable"] is False
    assert "GITHUB_ALLOW_WRITE=false" in res_issue["error"]["message"]

    # 2. create_issue_comment
    res_comment = create_issue_comment("octocat/Hello-World", 1, "A test comment")
    assert res_comment["ok"] is False
    assert res_comment["error"]["code"] == "permission_error"
    assert res_comment["error"]["retryable"] is False


def test_write_tools_succeed_when_allowed():
    """Verify write tools execute when GITHUB_ALLOW_WRITE=true."""
    settings.GITHUB_ALLOW_WRITE = True

    res = create_issue("octocat/Hello-World", "Allowed Issue", "Created safely")
    assert res["ok"] is True
    assert res["data"]["title"] == "Allowed Issue"
    assert "issues/" in res["data"]["html_url"]

    res_comment = create_issue_comment("octocat/Hello-World", res["data"]["number"], "New comment")
    assert res_comment["ok"] is True
    assert res_comment["data"]["body"] == "New comment"


# --- 2. Error Taxonomy & Guardrail 8 Tests ---

def test_404_maps_to_not_found_structured_error():
    """Verify non-existent repository or issue produces structured 'not_found' error."""
    # 404 Repository
    res = get_repository("octocat/notfound-404-repo")
    assert res["ok"] is False
    assert res["error"]["code"] == "not_found"
    assert "not found" in res["error"]["message"].lower()

    # 404 Issue
    res_issue = get_issue("octocat/Hello-World", 9999)
    assert res_issue["ok"] is False
    assert res_issue["error"]["code"] == "not_found"


def test_rate_limit_maps_to_rate_limited_structured_error():
    """Verify rate limit error produces structured 'rate_limited' error."""
    res = get_repository("octocat/ratelimit-repo")
    assert res["ok"] is False
    assert res["error"]["code"] == "rate_limited"
    assert res["error"]["retryable"] is True


def test_agent_handles_error_without_crashing():
    """Verify GitHubAgent translates tool error into AgentResult(status='error')."""
    agent = GitHubAgent()
    res = agent.invoke("Show issues in octocat/notfound-repo")
    assert res.status == "error"
    assert res.error is not None
    assert res.error["code"] == "not_found"


# --- 3. Hard Rule 1: No Stdout in MCP Servers ---

def test_mcp_server_writes_zero_bytes_to_stdout(capsys):
    """Verify that calling GitHub MCP tools writes 0 bytes to sys.stdout."""
    # Call multiple MCP tools
    search_repositories("octocat")
    get_repository("octocat/Hello-World")
    list_issues("octocat/Hello-World")
    get_file_contents("octocat/Hello-World", "README.md")
    list_branches("octocat/Hello-World")
    get_user("octocat")

    # Capture stdout and stderr
    captured = capsys.readouterr()
    assert captured.out == "", f"Hard Rule 1 violated! Stdout captured: {captured.out!r}"


# --- 4. Tool Registry Inspection ---

def test_fastmcp_tool_registry():
    """Verify FastMCP tool registration matches SPEC."""
    tool_names = [t.name for t in github_server.mcp._tool_manager.list_tools()]
    expected_tools = {
        "search_repositories",
        "get_repository",
        "list_issues",
        "get_issue",
        "list_pull_requests",
        "get_pull_request",
        "list_commits",
        "get_file_contents",
        "search_code",
        "list_branches",
        "get_user",
        "create_issue",
        "create_issue_comment",
    }
    assert expected_tools.issubset(set(tool_names))


# --- 5. Regression Tests: Repository Search & Username Disambiguation ---

def test_search_repositories_by_username():
    """Verify 'search repositories of git abdullahnadeem215' searches user:abdullahnadeem215, not octocat."""
    agent = GitHubAgent()
    res = agent.invoke("search repositories of git abdullahnadeem215")

    assert res.status == "ok"
    assert res.data["user"] == "abdullahnadeem215"
    assert "abdullahnadeem215/Toothology" in res.summary
    assert "octocat/Hello-World" not in res.summary


def test_list_all_repositories_resolves_to_authenticated_user():
    """Verify 'all repos' resolves to the authenticated GitHub user rather than defaulting to octocat/Hello-World."""
    agent = GitHubAgent()
    res = agent.invoke("all repos")

    assert res.status == "ok"
    assert res.data["user"] == "octocat"
    assert "Found 1 repositories for authenticated user 'octocat'" in res.summary
    assert "octocat/Hello-World" in res.summary


def test_explicit_owner_repo_calls_get_repository():
    """Verify explicit owner/repo format continues to fetch specific repository details."""
    agent = GitHubAgent()
    res = agent.invoke("Get repository details for octocat/Hello-World")

    assert res.status == "ok"
    assert res.data["full_name"] == "octocat/Hello-World"
    assert res.data["stars"] == 2500
    assert "Repository: octocat/Hello-World" in res.summary
    assert "Stars: 2500" in res.summary


# --- 6. Multi-Turn Graph Context & State Preservation Tests ---

def test_search_repositories_stores_target_user_in_artifacts():
    """Verify 'search repositories of git abdullahnadeem215' stores target username in artifacts."""
    from langgraph.checkpoint.memory import MemorySaver
    from app.service import ChatService

    service = ChatService(checkpointer=MemorySaver())
    thread_id = "test_github_context_thread_1"

    res = service.send(thread_id, "search repositories of git abdullahnadeem215")
    assert res["status"] == "completed"
    assert "Toothology" in res["response"]
    assert "octocat" not in res["response"].lower()

    # Check graph checkpoint state directly
    graph_state = service.graph.get_state({"configurable": {"thread_id": thread_id}})
    artifacts = graph_state.values.get("artifacts", {})
    assert artifacts.get("github_target_user") == "abdullahnadeem215"


def test_followup_all_repos_uses_context_target_user():
    """Verify multi-turn flow: 'all repos' following 'search repos of abdullahnadeem215' searches user:abdullahnadeem215."""
    from langgraph.checkpoint.memory import MemorySaver
    from app.service import ChatService

    service = ChatService(checkpointer=MemorySaver())
    thread_id = "test_github_context_thread_2"

    # Turn 1: Target abdullahnadeem215
    res1 = service.send(thread_id, "search repositories of git abdullahnadeem215")
    assert res1["status"] == "completed"
    assert "Toothology" in res1["response"]

    # Turn 2: Follow-up 'all repos'
    res2 = service.send(thread_id, "all repos")
    assert res2["status"] == "completed"
    assert "abdullahnadeem215" in res2["response"]
    assert "Toothology" in res2["response"]
    assert "octocat" not in res2["response"].lower()


def test_fresh_all_repos_uses_authenticated_user():
    """Verify 'all repos' in a fresh conversation resolves to the authenticated user (octocat in test mock)."""
    from langgraph.checkpoint.memory import MemorySaver
    from app.service import ChatService

    service = ChatService(checkpointer=MemorySaver())
    thread_id = "test_github_context_thread_3"

    res = service.send(thread_id, "all repos")
    assert res["status"] == "completed"
    assert "octocat" in res["response"]
    assert "Hello-World" in res["response"]
    assert "abdullahnadeem215" not in res["response"]


def test_unrelated_conversation_does_not_affect_all_repos():
    """Verify intervening unrelated turn (chitchat) clears the target user context for 'all repos'."""
    from langgraph.checkpoint.memory import MemorySaver
    from app.service import ChatService

    service = ChatService(checkpointer=MemorySaver())
    thread_id = "test_github_context_thread_4"

    # Turn 1: Target abdullahnadeem215
    res1 = service.send(thread_id, "search repositories of git abdullahnadeem215")
    assert res1["status"] == "completed"
    assert "Toothology" in res1["response"]

    # Turn 2: Unrelated chitchat
    res2 = service.send(thread_id, "Hi, who are you?")
    assert res2["status"] == "completed"
    assert "assistant" in res2["response"].lower()

    # Turn 3: 'all repos' must NOT use abdullahnadeem215
    res3 = service.send(thread_id, "all repos")
    assert res3["status"] == "completed"
    assert "octocat" in res3["response"]
    assert "abdullahnadeem215" not in res3["response"]


def test_user_with_47_repositories_returns_all_47(setup_github_mock):
    """Verify that a user with 47 mock repositories returns all 47 repositories."""
    fake_client = setup_github_mock
    for i in range(1, 48):
        name = f"mockuser47/repo-{i:02d}"
        fake_client.repos[name] = {
            "name": f"repo-{i:02d}",
            "full_name": name,
            "description": f"Repository {i}",
            "html_url": f"https://github.com/{name}",
            "stargazers_count": i,
            "forks_count": 0,
        }
    agent = GitHubAgent()
    res = agent.invoke("search repositories of git mockuser47")
    assert res.status == "ok"
    assert len(res.data["repositories"]) == 47
    assert "Found 47 repositories" in res.summary
    assert "mockuser47/repo-01" in res.summary
    assert "mockuser47/repo-47" in res.summary


def test_pagination_combines_multiple_pages_correctly():
    """Verify that search_repositories combines multiple pages into a single result."""
    class MockPaginatedClient:
        def search_repositories(self, query: str):
            page1 = [
                {"name": f"repo-{i:02d}", "full_name": f"org/repo-{i:02d}", "html_url": f"https://github.com/org/repo-{i:02d}"}
                for i in range(1, 31)
            ]
            page2 = [
                {"name": f"repo-{i:02d}", "full_name": f"org/repo-{i:02d}", "html_url": f"https://github.com/org/repo-{i:02d}"}
                for i in range(31, 48)
            ]
            for p in (page1, page2):
                for item in p:
                    yield item

    set_github_client(MockPaginatedClient())
    try:
        res = search_repositories("user:org", limit=100)
        assert res["ok"] is True
        assert res["data"]["count"] == 47
        assert len(res["data"]["repositories"]) == 47
        assert res["data"]["repositories"][0]["full_name"] == "org/repo-01"
        assert res["data"]["repositories"][46]["full_name"] == "org/repo-47"
    finally:
        set_github_client(None)


def test_no_duplicate_repositories_across_pages():
    """Verify that search_repositories deduplicates items across pages."""
    class MockDuplicatePageClient:
        def search_repositories(self, query: str):
            page1 = [
                {"name": "repo-01", "full_name": "org/repo-01", "html_url": "https://github.com/org/repo-01"},
                {"name": "repo-02", "full_name": "org/repo-02", "html_url": "https://github.com/org/repo-02"},
            ]
            page2 = [
                {"name": "repo-02", "full_name": "org/repo-02", "html_url": "https://github.com/org/repo-02"},
                {"name": "repo-03", "full_name": "org/repo-03", "html_url": "https://github.com/org/repo-03"},
            ]
            for p in (page1, page2):
                for item in p:
                    yield item

    set_github_client(MockDuplicatePageClient())
    try:
        res = search_repositories("user:org", limit=100)
        assert res["ok"] is True
        assert res["data"]["count"] == 3
        full_names = [r["full_name"] for r in res["data"]["repositories"]]
        assert full_names == ["org/repo-01", "org/repo-02", "org/repo-03"]
    finally:
        set_github_client(None)


def test_existing_ten_or_fewer_behavior_remains_correct(setup_github_mock):
    """Verify that searches returning 10 or fewer results remain unaffected."""
    agent = GitHubAgent()
    res = agent.invoke("search repositories of git abdullahnadeem215")
    assert res.status == "ok"
    assert len(res.data["repositories"]) == 1
    assert "Found 1 repositories" in res.summary
    assert "abdullahnadeem215/Toothology" in res.summary


def test_all_repos_multi_turn_returns_all_target_repos(setup_github_mock):
    """Verify follow-up 'all repos' returns all 47 repositories for preserved target username."""
    fake_client = setup_github_mock
    for i in range(1, 48):
        name = f"user47/project-{i:02d}"
        fake_client.repos[name] = {
            "name": f"project-{i:02d}",
            "full_name": name,
            "description": f"Project {i}",
            "html_url": f"https://github.com/{name}",
            "stargazers_count": i,
            "forks_count": 0,
        }

    from langgraph.checkpoint.memory import MemorySaver
    from app.service import ChatService

    service = ChatService(checkpointer=MemorySaver())
    thread_id = "test_github_pagination_multi_turn"

    # Turn 1: Search repos of user47
    res1 = service.send(thread_id, "search repositories of git user47")
    assert res1["status"] == "completed"
    assert "Found 47 repositories" in res1["response"]

    # Turn 2: Follow-up 'all repos'
    res2 = service.send(thread_id, "all repos")
    assert res2["status"] == "completed"
    assert "user47" in res2["response"]
    assert "Found 47 repositories" in res2["response"]
    assert "user47/project-01" in res2["response"]
    assert "user47/project-47" in res2["response"]
    assert "octocat" not in res2["response"].lower()


