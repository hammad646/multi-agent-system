"""Fake client skeletons for testing and MOCK_MODE=true.

Guarantees 100% offline testing without hitting Gmail, Calendar, GitHub, or Pinecone.
"""
from datetime import datetime, timezone
import hashlib
import re
from typing import Any
import uuid



class FakeLLM:
    """Fake LLM for deterministic testing and router/agent mocks."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = responses or []
        self.call_history: list[dict[str, Any]] = []

    def with_structured_output(self, schema: Any) -> "FakeLLM":
        """Mock langchain structured output binding."""
        self._schema = schema
        return self

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        """Asynchronously return queued response or default schema instance."""
        self.call_history.append({"messages": messages, "kwargs": kwargs})
        if self.responses:
            return self.responses.pop(0)
        if hasattr(self, "_schema") and self._schema is not None:
            # Return dummy default instance if schema is a Pydantic model
            try:
                return self._schema.model_validate(
                    {"intent": "chitchat", "reasoning": "mock response"}
                )
            except Exception:
                pass
        return "Fake LLM response"

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """Synchronously return queued response or default schema instance."""
        self.call_history.append({"messages": messages, "kwargs": kwargs})
        if self.responses:
            return self.responses.pop(0)
        return "Fake LLM response"


class FakePineconeIndex:
    """Fake Pinecone vector index with in-memory storage."""

    def __init__(self) -> None:
        # namespace -> {vec_id: {"id": str, "values": list[float], "metadata": dict}}
        self.vectors: dict[str, dict[str, dict[str, Any]]] = {}

    def upsert(
        self,
        vectors: list[tuple[str, list[float], dict[str, Any]] | dict[str, Any]],
        namespace: str = "",
    ) -> dict[str, int]:
        """Upsert vectors into in-memory store."""
        if namespace not in self.vectors:
            self.vectors[namespace] = {}

        count = 0
        for item in vectors:
            if isinstance(item, dict):
                vec_id = item["id"]
                vals = item.get("values", [])
                meta = item.get("metadata", {})
            else:
                vec_id, vals, meta = item
            self.vectors[namespace][vec_id] = {
                "id": vec_id,
                "values": vals,
                "metadata": meta,
            }
            count += 1
        return {"upserted_count": count}

    def query(
        self,
        vector: list[float] | None = None,
        top_k: int = 5,
        namespace: str = "",
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Query vectors with basic metadata filtering and cosine similarity."""
        namespaces_to_search = (
            [namespace] if (namespace and namespace in self.vectors) else list(self.vectors.keys())
        )
        candidates = []
        for ns in namespaces_to_search:
            for vec_id, data in self.vectors.get(ns, {}).items():
                meta = data.get("metadata", {})
                if filter:
                    match = True
                    for k, v in filter.items():
                        if meta.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                score = 0.85
                if vector and data.get("values") and len(vector) == len(data["values"]):
                    dot = sum(a * b for a, b in zip(vector, data["values"]))
                    n1 = sum(a * a for a in vector) ** 0.5
                    n2 = sum(b * b for b in data["values"]) ** 0.5
                    score = dot / (n1 * n2) if (n1 > 0 and n2 > 0) else 0.0

                candidates.append({
                    "id": vec_id,
                    "score": float(score),
                    "metadata": meta if include_metadata else {},
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": candidates[:top_k], "namespace": namespace}

    def delete(
        self,
        ids: list[str] | None = None,
        delete_all: bool = False,
        namespace: str = "",
    ) -> dict[str, Any]:
        """Delete vectors from in-memory store."""
        if namespace in self.vectors:
            if delete_all:
                self.vectors[namespace].clear()
            elif ids:
                for vec_id in ids:
                    self.vectors[namespace].pop(vec_id, None)
        elif delete_all and not namespace:
            self.vectors.clear()
        return {"status": "ok"}


class FakeInference:
    """Fake Pinecone inference for reranking and embeddings."""

    def embed(
        self,
        model: str,
        inputs: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> Any:
        """Generate deterministic dense embeddings reflecting semantic word overlap."""
        class EmbeddingItem:
            def __init__(self, values: list[float]) -> None:
                self.values = values

        class EmbedResponse:
            def __init__(self, data: list[Any]) -> None:
                self.data = data

        results = []
        dim = 256
        for text in inputs:
            vec = [0.0] * dim
            words = re.findall(r"\w+", text.lower())
            if not words:
                results.append(EmbeddingItem([0.0] * dim))
                continue

            for w in words:
                # Hash word to index and sign for feature hashing
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vec[idx] += sign

            # Normalize vector to unit length
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            results.append(EmbeddingItem(vec))

        return EmbedResponse(results)


    def rerank(
        self,
        model: str,
        query: str,
        documents: list[dict[str, Any] | str],
        top_n: int = 5,
        return_documents: bool = True,
        parameters: dict[str, Any] | None = None,
    ) -> Any:
        stopwords = {
            "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or",
            "is", "are", "was", "were", "what", "where", "how", "who", "which",
            "with", "it", "this", "that", "do", "does", "did", "can", "could",
        }
        all_query_words = re.findall(r"\w+", query.lower())
        query_words = set(all_query_words) - stopwords
        if not query_words:
            query_words = set(all_query_words)

        results = []
        for i, doc in enumerate(documents):
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            doc_words = set(re.findall(r"\w+", text.lower())) - stopwords
            overlap = len(query_words.intersection(doc_words))
            score = (overlap / len(query_words)) if query_words else 0.0
            res_item: dict[str, Any] = {"index": i, "score": float(score)}
            if return_documents:
                res_item["document"] = doc if isinstance(doc, dict) else {"text": text}
            results.append(res_item)

        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_n]


        class RerankResult:
            def __init__(self, data: list[Any]) -> None:
                self.data = [
                    type("RerankItem", (), item)() for item in data
                ]

        return RerankResult(top_results)


class FakePinecone:
    """Fake Pinecone client."""

    def __init__(self, api_key: str = "mock-key") -> None:
        self.api_key = api_key
        self.indexes: dict[str, FakePineconeIndex] = {}
        self.inference = FakeInference()

    def Index(self, name: str) -> FakePineconeIndex:
        """Get or create in-memory index."""
        if name not in self.indexes:
            self.indexes[name] = FakePineconeIndex()
        return self.indexes[name]



class FakeGitHubClient:
    """Fake GitHub client providing mock repository and issue operations."""

    def __init__(self, token: str = "mock-token") -> None:
        self.token = token
        self.users: dict[str, dict[str, Any]] = {
            "octocat": {
                "login": "octocat",
                "name": "The Octocat",
                "bio": "GitHub mascot and bot builder",
                "public_repos": 8,
                "followers": 9999,
                "html_url": "https://github.com/octocat",
            },
            "abdullahnadeem215": {
                "login": "abdullahnadeem215",
                "name": "Abdullah Nadeem",
                "bio": "AI & Software Engineer",
                "public_repos": 10,
                "followers": 50,
                "html_url": "https://github.com/abdullahnadeem215",
            },
        }
        self.repos: dict[str, dict[str, Any]] = {
            "abdullahnadeem215/Toothology": {
                "name": "Toothology",
                "full_name": "abdullahnadeem215/Toothology",
                "description": "Dental clinic management platform",
                "html_url": "https://github.com/abdullahnadeem215/Toothology",
                "stargazers_count": 5,
                "forks_count": 1,
                "default_branch": "main",
                "issues": [],
                "files": {},
                "branches": ["main"],
                "commits": [],
            },
            "test/repo": {
                "name": "repo",
                "full_name": "test/repo",
                "description": "A test repository",
                "html_url": "https://github.com/test/repo",
                "stargazers_count": 10,
                "forks_count": 2,
                "default_branch": "main",
                "issues": [
                    {
                        "number": 1,
                        "title": "Initial issue",
                        "body": "Need tests",
                        "state": "open",
                        "html_url": "https://github.com/test/repo/issues/1",
                        "comments": [],
                    }
                ],
                "files": {
                    "README.md": "# Test Repo\nThis is a mock repository.",
                },
                "branches": ["main", "feature"],
                "commits": [{"sha": "mocksha123", "message": "Initial commit", "author": "tester"}],
            },
            "octocat/Hello-World": {
                "name": "Hello-World",
                "full_name": "octocat/Hello-World",
                "description": "My first repository on GitHub!",
                "html_url": "https://github.com/octocat/Hello-World",
                "stargazers_count": 2500,
                "forks_count": 1200,
                "default_branch": "main",

                "issues": [
                    {
                        "number": 1,
                        "title": "Bug in greeting message",
                        "body": "The hello world greeting is missing an exclamation mark.",
                        "state": "open",
                        "html_url": "https://github.com/octocat/Hello-World/issues/1",
                        "comments": ["Working on a fix."],
                    },
                    {
                        "number": 2,
                        "title": "Add multilingual support",
                        "body": "Support Spanish, French, and Urdu.",
                        "state": "open",
                        "html_url": "https://github.com/octocat/Hello-World/issues/2",
                        "comments": [],
                    },
                ],
                "pull_requests": [
                    {
                        "number": 3,
                        "title": "Fix punctuation in greeting",
                        "body": "Resolves issue #1 by adding exclamation mark.",
                        "state": "open",
                        "html_url": "https://github.com/octocat/Hello-World/pull/3",
                    }
                ],
                "files": {
                    "README.md": "# Hello-World\n\nThis is a sample repository demonstration.",
                    "main.py": "print('Hello World!')\n",
                },
                "branches": ["main", "dev", "feature-greeting"],
                "commits": [
                    {"sha": "c0ffee1", "message": "Initial commit", "author": "octocat"},
                    {"sha": "c0ffee2", "message": "Update README.md", "author": "octocat"},
                ],
            }
        }

    def _check_error_triggers(self, repo_name: str) -> None:
        """Simulate external API errors for testing."""
        if "404" in repo_name or "notfound" in repo_name:
            raise Exception("404 Not Found: Repository does not exist")
        if "ratelimit" in repo_name or "403" in repo_name:
            raise Exception("403 API rate limit exceeded for user ID")
        if "500" in repo_name or "servererror" in repo_name:
            raise Exception("500 Internal Server Error")

    def search_repositories(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        self._check_error_triggers(query)
        q = query.lower().strip()
        results = []
        seen = set()
        if q.startswith("user:"):
            target_user = q.split("user:")[1].strip()
            for name, data in self.repos.items():
                if name.lower().startswith(f"{target_user}/") or target_user in name.lower():
                    fn = data.get("full_name", name)
                    if fn not in seen:
                        seen.add(fn)
                        results.append(data)
        else:
            for name, data in self.repos.items():
                if q in name.lower() or q in data.get("description", "").lower():
                    fn = data.get("full_name", name)
                    if fn not in seen:
                        seen.add(fn)
                        results.append(data)
        return results[:limit]

    def get_repo(self, repo_name: str) -> dict[str, Any]:
        self._check_error_triggers(repo_name)
        if repo_name not in self.repos:
            raise Exception(f"404 Not Found: Repository {repo_name} not found")
        return self.repos[repo_name]

    def list_issues(self, repo_name: str, state: str = "open", limit: int = 10) -> list[dict[str, Any]]:
        repo = self.get_repo(repo_name)
        issues = repo.get("issues", [])
        if state != "all":
            issues = [i for i in issues if i.get("state") == state]
        return issues[:limit]

    def get_issue(self, repo_name: str, number: int) -> dict[str, Any]:
        repo = self.get_repo(repo_name)
        for issue in repo.get("issues", []):
            if issue.get("number") == number:
                return issue
        raise Exception(f"404 Not Found: Issue #{number} not found in {repo_name}")

    def list_pull_requests(self, repo_name: str, state: str = "open", limit: int = 10) -> list[dict[str, Any]]:
        repo = self.get_repo(repo_name)
        prs = repo.get("pull_requests", [])
        if state != "all":
            prs = [p for p in prs if p.get("state") == state]
        return prs[:limit]

    def get_pull_request(self, repo_name: str, number: int) -> dict[str, Any]:
        repo = self.get_repo(repo_name)
        for pr in repo.get("pull_requests", []):
            if pr.get("number") == number:
                return pr
        raise Exception(f"404 Not Found: PR #{number} not found in {repo_name}")

    def list_commits(self, repo_name: str, limit: int = 10) -> list[dict[str, Any]]:
        repo = self.get_repo(repo_name)
        return repo.get("commits", [])[:limit]

    def get_file_contents(self, repo_name: str, path: str = "README.md", ref: str = "main") -> dict[str, Any]:
        repo = self.get_repo(repo_name)
        files = repo.get("files", {})
        if path not in files:
            raise Exception(f"404 Not Found: File '{path}' not found in {repo_name}")
        return {
            "path": path,
            "content": files[path],
            "encoding": "utf-8",
            "size": len(files[path]),
        }

    def search_code(self, repo_name: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        repo = self.get_repo(repo_name)
        q = query.lower()
        matches = []
        for file_path, content in repo.get("files", {}).items():
            if q in content.lower():
                matches.append({"path": file_path, "repository": repo_name})
        return matches[:limit]

    def list_branches(self, repo_name: str) -> list[str]:
        repo = self.get_repo(repo_name)
        return repo.get("branches", [])

    def get_user(self, username: str | None = None) -> dict[str, Any]:
        u = username or "octocat"
        if u not in self.users:
            raise Exception(f"404 Not Found: User '{u}' not found")
        return self.users[u]

    def create_issue(self, repo_name: str, title: str, body: str) -> dict[str, Any]:
        repo = self.get_repo(repo_name)
        num = len(repo.get("issues", [])) + len(repo.get("pull_requests", [])) + 1
        new_issue = {
            "number": num,
            "title": title,
            "body": body,
            "state": "open",
            "html_url": f"https://github.com/{repo_name}/issues/{num}",
            "comments": [],
        }
        repo.setdefault("issues", []).append(new_issue)
        return new_issue

    def create_issue_comment(self, repo_name: str, number: int, comment: str) -> dict[str, Any]:
        issue = self.get_issue(repo_name, number)
        issue.setdefault("comments", []).append(comment)
        return {"id": len(issue["comments"]), "body": comment}



class FakeCalendarClient:
    """Fake Google Calendar client for mock event management and offline testing."""

    def __init__(self) -> None:
        self.events: dict[str, dict[str, Any]] = {}

    def _parse_iso(self, iso_str: str) -> datetime:
        """Parse ISO string and strictly enforce timezone awareness."""
        if not iso_str:
            raise ValueError("Datetime string cannot be empty")
        # Replace Z with +00:00 for fromisoformat compatibility
        cleaned = iso_str.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
        except Exception as exc:
            raise ValueError(f"Invalid ISO 8601 datetime format: {iso_str}") from exc

        if dt.tzinfo is None:
            raise ValueError(
                "Naive datetimes are rejected per SPEC.md guardrails. Timezone offset is required."
            )
        return dt

    def list_events(
        self, time_min: str | None = None, time_max: str | None = None, max_results: int = 25
    ) -> list[dict[str, Any]]:
        events = list(self.events.values())
        if time_min:
            t_min = self._parse_iso(time_min)
            events = [e for e in events if self._parse_iso(e["start"]["dateTime"]) >= t_min]
        if time_max:
            t_max = self._parse_iso(time_max)
            events = [e for e in events if self._parse_iso(e["start"]["dateTime"]) <= t_max]
        events.sort(key=lambda e: self._parse_iso(e["start"]["dateTime"]))
        return events[:max_results]

    def get_event(self, event_id: str) -> dict[str, Any]:
        if event_id not in self.events:
            raise Exception(f"404 Not Found: Event {event_id} does not exist")
        return self.events[event_id]

    def find_conflicts(self, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
        """Find existing events overlapping with the requested interval."""
        target_start = self._parse_iso(start_iso)
        target_end = self._parse_iso(end_iso)
        conflicts = []
        for e in self.events.values():
            e_start = self._parse_iso(e["start"]["dateTime"])
            e_end = self._parse_iso(e["end"]["dateTime"])
            # Overlap condition: start < other_end and end > other_start
            if target_start < e_end and target_end > e_start:
                conflicts.append(e)
        return conflicts

    def find_free_slots(
        self, start_date: str, end_date: str, duration_minutes: int = 30
    ) -> list[dict[str, Any]]:
        """Calculate free slots between 09:00 and 17:00 excluding existing bookings."""
        # Simple simulated free slots generator for testing
        d_start = self._parse_iso(start_date)
        slots = []
        for hour in [10, 11, 14, 15, 16]:
            slot_start = d_start.replace(hour=hour, minute=0, second=0, microsecond=0)
            slot_end = slot_start + timedelta(minutes=duration_minutes)
            s_iso = slot_start.isoformat()
            e_iso = slot_end.isoformat()
            if not self.find_conflicts(s_iso, e_iso):
                slots.append({"start": s_iso, "end": e_iso, "duration_minutes": duration_minutes})
        return slots

    def create_event(
        self,
        summary: str,
        start_iso: str,
        end_iso: str,
        attendees: list[str] | None = None,
        description: str = "",
        add_meet_link: bool = False,
    ) -> dict[str, Any]:
        """Create a calendar event with strict naive datetime rejection and duplicate detection."""
        # Enforce timezone awareness
        self._parse_iso(start_iso)
        self._parse_iso(end_iso)

        # Duplicate detection by title + start time
        for existing in self.events.values():
            if (
                existing.get("summary", "").strip().lower() == summary.strip().lower()
                and existing.get("start", {}).get("dateTime") == start_iso
            ):
                return {
                    "duplicate": True,
                    "event": existing,
                    "message": f"Duplicate event '{summary}' already exists at {start_iso}",
                }

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "summary": summary,
            "start": {"dateTime": start_iso},
            "end": {"dateTime": end_iso},
            "attendees": [{"email": a} for a in (attendees or [])],
            "description": description,
            "status": "confirmed",
            "hangoutLink": "https://meet.google.com/abc-mock-meet" if add_meet_link else None,
            "duplicate": False,
        }
        self.events[event_id] = event
        return event

    def update_event(
        self,
        event_id: str,
        summary: str | None = None,
        start_iso: str | None = None,
        end_iso: str | None = None,
        attendees: list[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        event = self.get_event(event_id)
        if summary is not None:
            event["summary"] = summary
        if start_iso is not None:
            self._parse_iso(start_iso)
            event["start"]["dateTime"] = start_iso
        if end_iso is not None:
            self._parse_iso(end_iso)
            event["end"]["dateTime"] = end_iso
        if attendees is not None:
            event["attendees"] = [{"email": a} for a in attendees]
        if description is not None:
            event["description"] = description
        return event

    def delete_event(self, event_id: str) -> bool:
        if event_id not in self.events:
            raise Exception(f"404 Not Found: Event {event_id} does not exist")
        self.events.pop(event_id)
        return True



class FakeGmailClient:
    """Fake Gmail client for mock email sending, drafting, and searching."""

    def __init__(self) -> None:
        self.drafts: dict[str, dict[str, Any]] = {}
        self.messages: dict[str, dict[str, Any]] = {}
        self.sent_messages: list[dict[str, Any]] = []
        self.trashed_messages: set[str] = set()
        self.simulate_failure: Exception | None = None
        self.labels: list[dict[str, str]] = [
            {"id": "INBOX", "name": "INBOX", "type": "system"},
            {"id": "SENT", "name": "SENT", "type": "system"},
            {"id": "DRAFT", "name": "DRAFT", "type": "system"},
            {"id": "TRASH", "name": "TRASH", "type": "system"},
            {"id": "SPAM", "name": "SPAM", "type": "system"},
        ]

    def _check_simulation(self) -> None:
        if self.simulate_failure:
            exc = self.simulate_failure
            raise exc

    def search_emails(self, query: str = "", max_results: int = 10) -> list[dict[str, Any]]:
        self._check_simulation()
        q_lower = query.lower()
        results = []
        for msg in self.messages.values():
            if msg["id"] in self.trashed_messages:
                continue
            if not query or (
                q_lower in msg.get("subject", "").lower()
                or q_lower in msg.get("body", "").lower()
                or q_lower in str(msg.get("to", "")).lower()
                or q_lower in str(msg.get("from", "")).lower()
            ):
                results.append(msg)
        return results[:max_results]

    def read_email(self, message_id: str) -> dict[str, Any]:
        self._check_simulation()
        if message_id in self.messages:
            msg = self.messages[message_id]
            if message_id in self.trashed_messages:
                raise Exception(f"404 Not Found: Email {message_id} is in trash.")
            return msg
        # Also check sent messages
        for msg in self.sent_messages:
            if msg.get("id") == message_id:
                return msg
        raise Exception(f"404 Not Found: Email {message_id} does not exist.")

    def create_draft(
        self,
        to: str | list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> dict[str, Any]:
        self._check_simulation()
        draft_id = f"draft_{uuid.uuid4().hex[:12]}"
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        draft = {
            "id": draft_id,
            "message": {
                "id": msg_id,
                "to": to,
                "cc": cc or [],
                "bcc": bcc or [],
                "subject": subject,
                "body": body,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        self.drafts[draft_id] = draft
        return draft

    def list_drafts(self, max_results: int = 10) -> list[dict[str, Any]]:
        self._check_simulation()
        return list(self.drafts.values())[:max_results]

    def send_draft(self, draft_id: str) -> dict[str, Any]:
        self._check_simulation()
        if draft_id not in self.drafts:
            raise Exception(f"404 Not Found: Draft {draft_id} does not exist.")
        draft = self.drafts.pop(draft_id)
        msg_info = draft["message"]
        sent = {
            "id": msg_info["id"],
            "to": msg_info["to"],
            "cc": msg_info.get("cc", []),
            "bcc": msg_info.get("bcc", []),
            "subject": msg_info["subject"],
            "body": msg_info["body"],
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        self.sent_messages.append(sent)
        self.messages[sent["id"]] = sent
        return sent

    def send_email(
        self,
        to: str | list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        self._check_simulation()
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        msg = {
            "id": message_id,
            "to": to,
            "cc": cc or [],
            "bcc": bcc or [],
            "subject": subject,
            "body": body,
            "threadId": thread_id or f"thread_{uuid.uuid4().hex[:12]}",
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        self.sent_messages.append(msg)
        self.messages[message_id] = msg
        return msg

    def reply_to_email(
        self,
        message_id: str,
        body: str,
        reply_all: bool = False,
    ) -> dict[str, Any]:
        self._check_simulation()
        original = self.read_email(message_id)
        orig_subject = original.get("subject", "")
        reply_subject = orig_subject if orig_subject.lower().startswith("re:") else f"Re: {orig_subject}"
        reply_to = original.get("from") or original.get("to")
        cc = original.get("cc", []) if reply_all else []
        reply_id = f"msg_{uuid.uuid4().hex[:12]}"
        reply_msg = {
            "id": reply_id,
            "to": reply_to,
            "cc": cc,
            "subject": reply_subject,
            "body": body,
            "in_reply_to": message_id,
            "references": message_id,
            "threadId": original.get("threadId", original.get("id")),
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        self.sent_messages.append(reply_msg)
        self.messages[reply_id] = reply_msg
        return reply_msg

    def delete_email(self, message_id: str) -> dict[str, Any]:
        self._check_simulation()
        if message_id not in self.messages and not any(m["id"] == message_id for m in self.sent_messages):
            raise Exception(f"404 Not Found: Email {message_id} does not exist.")
        self.trashed_messages.add(message_id)
        return {"id": message_id, "status": "trashed"}

    def list_labels(self) -> list[dict[str, Any]]:
        self._check_simulation()
        return self.labels

