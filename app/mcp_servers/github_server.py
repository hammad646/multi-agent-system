"""GitHub FastMCP server per SPEC.md section 10.

Runs over stdio. Strictly writes logs to stderr, NEVER stdout.
Provides tools: search_repositories, get_repository, list_issues, get_issue,
list_pull_requests, get_pull_request, list_commits, get_file_contents,
search_code, list_branches, get_user, create_issue, create_issue_comment.
Read-only by default; write operations are blocked unless GITHUB_ALLOW_WRITE=true.
"""
import sys
from typing import Any
from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.errors import create_error_response
from app.logging import get_logger
from app.mcp_servers.common import mcp_error_handler, retry_external_call

logger = get_logger("mcp_servers.github")

mcp = FastMCP("github")

_github_client: Any = None


def get_github_client() -> Any:
    """Get active GitHub client (real PyGithub or FakeGitHubClient in mock mode)."""
    global _github_client
    if _github_client is not None:
        return _github_client

    if settings.MOCK_MODE or not settings.GITHUB_PAT:
        from app.testing.fakes import FakeGitHubClient
        _github_client = FakeGitHubClient(token=settings.GITHUB_PAT or "mock-pat")
        return _github_client

    try:
        from github import Github, Auth
        auth = Auth.Token(settings.GITHUB_PAT)
        _github_client = Github(auth=auth)
        return _github_client
    except Exception as exc:
        logger.error("github_client_init_failed", error=str(exc))
        from app.testing.fakes import FakeGitHubClient
        _github_client = FakeGitHubClient(token="mock-pat")
        return _github_client


def set_github_client(client: Any) -> None:
    """Explicitly inject a client (used in tests)."""
    global _github_client
    _github_client = client


@mcp.tool()
@mcp_error_handler("github")
def search_repositories(query: str, limit: int = 100) -> dict[str, Any]:
    """Search GitHub repositories by keyword or topic."""
    client = get_github_client()
    if hasattr(client, "search_repositories"):
        # Real PyGithub returns PaginatedList; fake returns list of dicts
        repos = client.search_repositories(query)
        results = []
        seen = set()
        max_limit = limit if (limit is not None and limit > 0) else 100
        for r in repos:
            if isinstance(r, dict):
                full_name = r.get("full_name") or r.get("name")
                if full_name and full_name in seen:
                    continue
                if full_name:
                    seen.add(full_name)
                results.append(r)
            else:
                full_name = getattr(r, "full_name", None) or getattr(r, "name", None)
                if full_name and full_name in seen:
                    continue
                if full_name:
                    seen.add(full_name)
                results.append(
                    {
                        "name": getattr(r, "name", ""),
                        "full_name": full_name or "",
                        "description": getattr(r, "description", None),
                        "html_url": getattr(r, "html_url", ""),
                        "stars": getattr(r, "stargazers_count", 0),
                        "forks": getattr(r, "forks_count", 0),
                    }
                )
            if len(results) >= max_limit:
                break
        return {"repositories": results, "count": len(results)}
    raise ValueError("search_repositories not supported by client")


@mcp.tool()
@mcp_error_handler("github")
def get_repository(repo: str) -> dict[str, Any]:
    """Get details, metadata, and star count for a repository (format: owner/repo)."""
    client = get_github_client()
    r = client.get_repo(repo)
    if isinstance(r, dict):
        res = dict(r)
        res.setdefault("stars", res.get("stargazers_count", 0))
        res.setdefault("forks", res.get("forks_count", 0))
        return res
    return {
        "name": r.name,
        "full_name": r.full_name,
        "description": r.description,
        "html_url": r.html_url,
        "stars": r.stargazers_count,
        "forks": r.forks_count,
        "default_branch": r.default_branch,
        "open_issues_count": r.open_issues_count,
    }


@mcp.tool()
@mcp_error_handler("github")
def list_issues(repo: str, state: str = "open", limit: int = 10) -> dict[str, Any]:
    """List issues in a repository (state: 'open', 'closed', 'all')."""
    client = get_github_client()
    if hasattr(client, "list_issues"):
        # Fake client
        issues = client.list_issues(repo, state=state, limit=limit)
        return {"issues": issues, "count": len(issues)}

    r = client.get_repo(repo)
    res = []
    for i in r.get_issues(state=state):
        if i.pull_request is None:  # Exclude pull requests
            res.append(
                {
                    "number": i.number,
                    "title": i.title,
                    "body": i.body,
                    "state": i.state,
                    "html_url": i.html_url,
                    "user": i.user.login if i.user else None,
                    "comments_count": i.comments,
                }
            )
            if len(res) >= limit:
                break
    return {"issues": res, "count": len(res)}


@mcp.tool()
@mcp_error_handler("github")
def get_issue(repo: str, number: int) -> dict[str, Any]:
    """Get full details of a specific issue in a repository."""
    client = get_github_client()
    if hasattr(client, "get_issue"):
        return client.get_issue(repo, number)

    r = client.get_repo(repo)
    i = r.get_issue(number)
    return {
        "number": i.number,
        "title": i.title,
        "body": i.body,
        "state": i.state,
        "html_url": i.html_url,
        "user": i.user.login if i.user else None,
        "comments": [c.body for c in i.get_comments()],
    }


@mcp.tool()
@mcp_error_handler("github")
def list_pull_requests(repo: str, state: str = "open", limit: int = 10) -> dict[str, Any]:
    """List pull requests in a repository (state: 'open', 'closed', 'all')."""
    client = get_github_client()
    if hasattr(client, "list_pull_requests"):
        prs = client.list_pull_requests(repo, state=state, limit=limit)
        return {"pull_requests": prs, "count": len(prs)}

    r = client.get_repo(repo)
    res = []
    for p in r.get_pulls(state=state):
        res.append(
            {
                "number": p.number,
                "title": p.title,
                "state": p.state,
                "html_url": p.html_url,
                "user": p.user.login if p.user else None,
            }
        )
        if len(res) >= limit:
            break
    return {"pull_requests": res, "count": len(res)}


@mcp.tool()
@mcp_error_handler("github")
def get_pull_request(repo: str, number: int) -> dict[str, Any]:
    """Get details of a specific pull request."""
    client = get_github_client()
    if hasattr(client, "get_pull_request"):
        return client.get_pull_request(repo, number)

    r = client.get_repo(repo)
    p = r.get_pull(number)
    return {
        "number": p.number,
        "title": p.title,
        "body": p.body,
        "state": p.state,
        "html_url": p.html_url,
        "diff_url": p.diff_url,
        "merged": p.merged,
    }


@mcp.tool()
@mcp_error_handler("github")
def list_commits(repo: str, limit: int = 10) -> dict[str, Any]:
    """List recent commits on the default branch of a repository."""
    client = get_github_client()
    if hasattr(client, "list_commits"):
        commits = client.list_commits(repo, limit=limit)
        return {"commits": commits, "count": len(commits)}

    r = client.get_repo(repo)
    res = []
    for c in r.get_commits():
        res.append(
            {
                "sha": c.sha,
                "message": c.commit.message,
                "author": c.commit.author.name if c.commit.author else None,
            }
        )
        if len(res) >= limit:
            break
    return {"commits": res, "count": len(res)}


@mcp.tool()
@mcp_error_handler("github")
def get_file_contents(repo: str, path: str = "README.md", ref: str = "main") -> dict[str, Any]:
    """Get the text content of a file in a repository."""
    client = get_github_client()
    if hasattr(client, "get_file_contents"):
        return client.get_file_contents(repo, path=path, ref=ref)

    r = client.get_repo(repo)
    target_ref = ref
    try:
        f = r.get_contents(path, ref=target_ref)
    except Exception:
        if target_ref == "main" and getattr(r, "default_branch", None) and r.default_branch != "main":
            f = r.get_contents(path, ref=r.default_branch)
        else:
            raise

    if isinstance(f, list):
        raise ValueError(f"'{path}' is a directory, not a file.")
    return {
        "path": f.path,
        "content": f.decoded_content.decode("utf-8", errors="replace"),
        "size": f.size,
    }


@mcp.tool()
@mcp_error_handler("github")
def search_code(repo: str, query: str, limit: int = 10) -> dict[str, Any]:
    """Search code within a specific repository."""
    client = get_github_client()
    if hasattr(client, "search_code"):
        matches = client.search_code(repo, query=query, limit=limit)
        return {"matches": matches, "count": len(matches)}

    full_query = f"{query} repo:{repo}"
    code_items = list(client.search_code(full_query))[:limit]
    return {
        "matches": [
            {"path": item.path, "repository": repo, "html_url": item.html_url}
            for item in code_items
        ],
        "count": len(code_items),
    }


@mcp.tool()
@mcp_error_handler("github")
def list_branches(repo: str) -> dict[str, Any]:
    """List branch names in a repository."""
    client = get_github_client()
    if hasattr(client, "list_branches"):
        branches = client.list_branches(repo)
        return {"branches": branches, "count": len(branches)}

    r = client.get_repo(repo)
    branches = [b.name for b in r.get_branches()]
    return {"branches": branches, "count": len(branches)}


@mcp.tool()
@mcp_error_handler("github")
def get_user(username: str | None = None) -> dict[str, Any]:
    """Get public GitHub user profile details."""
    client = get_github_client()
    u = client.get_user(username) if username else client.get_user()
    if isinstance(u, dict):
        return u
    return {
        "login": u.login,
        "name": u.name,
        "bio": u.bio,
        "public_repos": u.public_repos,
        "followers": u.followers,
        "following": u.following,
        "html_url": u.html_url,
    }


@mcp.tool()
@mcp_error_handler("github")
def create_issue(repo: str, title: str, body: str) -> dict[str, Any]:
    """Create a new issue in a repository. (Requires GITHUB_ALLOW_WRITE=true and approval)."""
    if not settings.GITHUB_ALLOW_WRITE:
        raise PermissionError(
            "GitHub write actions are disabled by policy (GITHUB_ALLOW_WRITE=false)."
        )

    client = get_github_client()
    if hasattr(client, "create_issue"):
        return client.create_issue(repo, title=title, body=body)

    r = client.get_repo(repo)
    issue = r.create_issue(title=title, body=body)
    return {
        "number": issue.number,
        "title": issue.title,
        "html_url": issue.html_url,
        "state": issue.state,
    }


@mcp.tool()
@mcp_error_handler("github")
def create_issue_comment(repo: str, number: int, comment: str) -> dict[str, Any]:
    """Add a comment to an existing issue. (Requires GITHUB_ALLOW_WRITE=true and approval)."""
    if not settings.GITHUB_ALLOW_WRITE:
        raise PermissionError(
            "GitHub write actions are disabled by policy (GITHUB_ALLOW_WRITE=false)."
        )

    client = get_github_client()
    if hasattr(client, "create_issue_comment"):
        return client.create_issue_comment(repo, number=number, comment=comment)

    r = client.get_repo(repo)
    i = r.get_issue(number)
    c = i.create_comment(comment)
    return {"id": c.id, "html_url": c.html_url, "body": c.body}


if __name__ == "__main__":
    # MCP servers communicate over stdio; logs go strictly to stderr/file
    mcp.run(transport="stdio")
