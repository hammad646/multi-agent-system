"""Database repositories implementing all SQL queries for the application.

All SQL logic lives strictly in this module. Agents, tools, and workers
interact exclusively via these repository classes using dependency injection.
"""
from datetime import datetime, timezone, timedelta
import json
from typing import Any
from sqlalchemy import select, update, delete
from sqlalchemy.orm import Session
from app.db.models import (
    DocumentModel,
    ChunkModel,
    ScheduledEmailModel,
    WorkflowModel,
    utc_now,
)


class DocumentRepository:
    """Repository for documents and their text chunks."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_document(
        self, document: DocumentModel, chunks: list[ChunkModel] | None = None
    ) -> DocumentModel:
        """Add a document and its chunks in a single transaction."""
        self.session.add(document)
        if chunks:
            for chunk in chunks:
                chunk.document_id = document.document_id
                self.session.add(chunk)
        self.session.commit()
        self.session.refresh(document)
        return document

    def get_document(self, document_id: str) -> DocumentModel | None:
        """Retrieve document by ID."""
        stmt = select(DocumentModel).where(DocumentModel.document_id == document_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_document_by_sha256(self, sha256: str) -> DocumentModel | None:
        """Retrieve document by SHA-256 hash to detect duplicates."""
        stmt = select(DocumentModel).where(DocumentModel.sha256 == sha256)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_documents(self) -> list[DocumentModel]:
        """List all ingested documents."""
        stmt = select(DocumentModel).order_by(DocumentModel.uploaded_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and cascade-delete its chunks."""
        doc = self.get_document(document_id)
        if doc is None:
            return False
        # Explicit delete to guarantee foreign key deletion across all SQLite/Postgres setups
        self.session.execute(
            delete(ChunkModel).where(ChunkModel.document_id == document_id)
        )
        self.session.delete(doc)
        self.session.commit()
        return True

    def get_chunks_for_document(self, document_id: str) -> list[ChunkModel]:
        """Retrieve all text chunks belonging to a document."""
        stmt = (
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.page.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_all_chunks(
        self, filter_doc_ids: list[str] | None = None
    ) -> list[ChunkModel]:
        """Retrieve chunks across documents, optionally filtered by document IDs."""
        stmt = select(ChunkModel)
        if filter_doc_ids:
            stmt = stmt.where(ChunkModel.document_id.in_(filter_doc_ids))
        return list(self.session.execute(stmt).scalars().all())


class ScheduledEmailRepository:
    """Repository for scheduled emails with atomic claim and worker support."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, email: ScheduledEmailModel) -> ScheduledEmailModel:
        """Persist a new scheduled email in pending status."""
        self.session.add(email)
        self.session.commit()
        self.session.refresh(email)
        return email

    def get(self, email_id: str) -> ScheduledEmailModel | None:
        """Get scheduled email by ID."""
        stmt = select(ScheduledEmailModel).where(ScheduledEmailModel.id == email_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_pending(
        self, due_before: datetime | None = None
    ) -> list[ScheduledEmailModel]:
        """List all pending emails, optionally filtered by scheduled_at <= due_before."""
        stmt = select(ScheduledEmailModel).where(
            ScheduledEmailModel.status == "pending"
        )
        if due_before:
            stmt = stmt.where(ScheduledEmailModel.scheduled_at <= due_before)
        stmt = stmt.order_by(ScheduledEmailModel.scheduled_at.asc())
        return list(self.session.execute(stmt).scalars().all())

    def list_all(
        self, user_id: str | None = None
    ) -> list[ScheduledEmailModel]:
        """List all scheduled emails, optionally filtered by user_id."""
        stmt = select(ScheduledEmailModel)
        if user_id:
            stmt = stmt.where(ScheduledEmailModel.user_id == user_id)
        stmt = stmt.order_by(ScheduledEmailModel.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def claim_pending(self, email_id: str) -> bool:
        """Atomically claim a pending email for sending.

        Executes:
        UPDATE scheduled_emails
        SET status='sending', attempts=attempts+1
        WHERE id=:id AND status='pending'

        Returns True if exactly one row changed, False otherwise.
        Guarantees that multiple concurrent workers cannot send the same email.
        """
        stmt = (
            update(ScheduledEmailModel)
            .where(
                ScheduledEmailModel.id == email_id,
                ScheduledEmailModel.status == "pending",
            )
            .values(
                status="sending",
                attempts=ScheduledEmailModel.attempts + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = self.session.execute(stmt)
        self.session.commit()
        return bool(result.rowcount == 1)

    def update_status(
        self,
        email_id: str,
        status: str,
        sent_at: datetime | None = None,
        last_error: str | None = None,
        gmail_message_id: str | None = None,
    ) -> bool:
        """Update email dispatch status and diagnostic details."""
        values: dict[str, Any] = {"status": status}
        if sent_at is not None:
            values["sent_at"] = sent_at
        if last_error is not None:
            values["last_error"] = last_error
        if gmail_message_id is not None:
            values["gmail_message_id"] = gmail_message_id

        stmt = (
            update(ScheduledEmailModel)
            .where(ScheduledEmailModel.id == email_id)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        result = self.session.execute(stmt)
        self.session.commit()
        return bool(result.rowcount == 1)

    def cancel(self, email_id: str) -> bool:
        """Cancel a pending scheduled email."""
        stmt = (
            update(ScheduledEmailModel)
            .where(
                ScheduledEmailModel.id == email_id,
                ScheduledEmailModel.status == "pending",
            )
            .values(status="cancelled")
            .execution_options(synchronize_session=False)
        )
        result = self.session.execute(stmt)
        self.session.commit()
        return bool(result.rowcount == 1)

    def delete(self, email_id: str) -> bool:
        """Permanently delete a scheduled email record."""
        stmt = delete(ScheduledEmailModel).where(ScheduledEmailModel.id == email_id)
        result = self.session.execute(stmt)
        self.session.commit()
        return bool(result.rowcount == 1)

    def reset_stuck_sending(self, threshold_seconds: int = 300) -> int:
        """Reset emails stuck in 'sending' status back to 'pending'.

        Handles cases where a worker crashed or was killed mid-send.
        If threshold_seconds <= 0, resets all sending rows.
        """
        stmt = update(ScheduledEmailModel).where(ScheduledEmailModel.status == "sending")
        if threshold_seconds > 0:
            stuck_cutoff = utc_now() - timedelta(seconds=threshold_seconds)
            stmt = stmt.where(ScheduledEmailModel.created_at <= stuck_cutoff)

        stmt = stmt.values(status="pending").execution_options(synchronize_session=False)
        result = self.session.execute(stmt)
        self.session.commit()
        return int(result.rowcount or 0)


class WorkflowRepository:
    """Repository for persisting and tracking workflow executions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        workflow_id: str,
        workflow_type: str,
        steps: dict[str, Any] | list[Any] | str = "{}",
    ) -> WorkflowModel:
        """Create a new workflow tracking record."""
        steps_str = steps if isinstance(steps, str) else json.dumps(steps)
        now = utc_now()
        wf = WorkflowModel(
            id=workflow_id,
            workflow_type=workflow_type,
            status="pending",
            steps=steps_str,
            created_at=now,
            updated_at=now,
        )
        self.session.add(wf)
        self.session.commit()
        self.session.refresh(wf)
        return wf

    def get(self, workflow_id: str) -> WorkflowModel | None:
        """Get workflow by ID."""
        stmt = select(WorkflowModel).where(WorkflowModel.id == workflow_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def update_status(
        self,
        workflow_id: str,
        status: str,
        steps: dict[str, Any] | list[Any] | str | None = None,
    ) -> WorkflowModel | None:
        """Update workflow status and optional step execution progress."""
        wf = self.get(workflow_id)
        if wf is None:
            return None
        wf.status = status
        wf.updated_at = utc_now()
        if steps is not None:
            wf.steps = steps if isinstance(steps, str) else json.dumps(steps)
        self.session.commit()
        self.session.refresh(wf)
        return wf

    def list_workflows(
        self, workflow_type: str | None = None
    ) -> list[WorkflowModel]:
        """List workflows, optionally filtered by type."""
        stmt = select(WorkflowModel)
        if workflow_type:
            stmt = stmt.where(WorkflowModel.workflow_type == workflow_type)
        stmt = stmt.order_by(WorkflowModel.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())
