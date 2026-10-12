"""Persistent SQLite case management system for financial anomaly investigations.

Provides ACID-compliant persistent storage for investigation cases,
analyst notes, transaction linkages, and tamper-evident audit history.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("finguard.investigation.case_manager")


class CaseManager:
    """Thread-safe, lightweight SQLite case manager for compliance investigations."""

    VALID_STATUSES = [
        "OPEN",
        "INVESTIGATING",
        "PENDING_CUSTOMER_CONTACT",
        "ESCALATED_AML",
        "CLOSED_FALSE_POSITIVE",
        "CLOSED_CONFIRMED_ANOMALY"
    ]

    VALID_PRIORITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initializes database connection and creates tables if not present."""
        if db_path is None:
            self.db_path = settings.DEFAULT_CASES_DB_PATH
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with row_factory set for dictionary access."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_database(self) -> None:
        """Initializes schema tables and indexes."""
        with self._get_connection() as conn:
            # 1. Cases Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    priority TEXT NOT NULL DEFAULT 'HIGH',
                    assigned_analyst TEXT NOT NULL DEFAULT 'Compliance-Desk',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    linked_transactions TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT ''
                )
            """)

            # 2. Case Notes Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS case_notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    author TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
                )
            """)

            # 3. Case History / Audit Trail Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS case_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    author TEXT NOT NULL,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
                )
            """)

            # 4. Alerts Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    transaction_id TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    applied_threshold REAL NOT NULL,
                    prediction TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    amount REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'NEW',
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    linked_case_id TEXT,
                    supporting_info TEXT NOT NULL DEFAULT '{}'
                )
            """)

            # 5. User Configurable Settings Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Indexes for high performance lookup
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases (priority)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_case_notes_case ON case_notes (case_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_case_history_case ON case_history (case_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_tx ON alerts (transaction_id)")
            conn.commit()

        logger.info(f"Initialized CaseManager SQLite schema at: {self.db_path}")

    @staticmethod
    def _now_iso() -> str:
        """Returns current UTC ISO-formatted timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _generate_case_id(self) -> str:
        """Generates consecutive sequential case identifier, e.g. CASE-2026-0001."""
        year = datetime.now().year
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM cases WHERE case_id LIKE ?",
                (f"CASE-{year}-%",)
            )
            count = cursor.fetchone()[0]
        return f"CASE-{year}-{count + 1:04d}"

    def create_case(
        self,
        title: str,
        priority: str = "HIGH",
        assigned_analyst: str = "Compliance-Analyst",
        linked_transactions: Optional[List[str]] = None,
        summary: str = "",
        initial_note: Optional[str] = None
    ) -> str:
        """Creates a new forensic case and logs initial audit record.

        Args:
            title: Descriptive headline for the investigation case.
            priority: Priority rating ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW').
            assigned_analyst: Identifier of officer assigned.
            linked_transactions: List of transaction IDs associated.
            summary: Contextual synopsis of the anomaly.
            initial_note: Optional preliminary investigation memo.

        Returns:
            The created case_id.
        """
        priority = priority.upper() if priority.upper() in self.VALID_PRIORITIES else "HIGH"
        txns = linked_transactions or []
        case_id = self._generate_case_id()
        now = self._now_iso()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO cases (case_id, title, status, priority, assigned_analyst, created_at, updated_at, linked_transactions, summary)
                VALUES (?, ?, 'OPEN', ?, ?, ?, ?, ?, ?)
                """,
                (case_id, title, priority, assigned_analyst, now, now, json.dumps(txns), summary)
            )

            # Record creation in audit history
            conn.execute(
                """
                INSERT INTO case_history (case_id, timestamp, action, details, author)
                VALUES (?, ?, 'CASE_CREATED', ?, ?)
                """,
                (case_id, now, f"Case opened with priority {priority} and {len(txns)} linked transaction(s).", assigned_analyst)
            )

            # Add initial note if provided
            if initial_note:
                conn.execute(
                    """
                    INSERT INTO case_notes (case_id, timestamp, author, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (case_id, now, assigned_analyst, initial_note)
                )

            conn.commit()

        logger.info(f"Created investigation case: {case_id} ('{title}')")
        return case_id

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a case record with its associated notes and audit trail."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
            row = cursor.fetchone()
            if row is None:
                return None

            case_data = dict(row)
            case_data["linked_transactions"] = json.loads(case_data.get("linked_transactions", "[]"))

            # Fetch notes
            cur_notes = conn.execute(
                "SELECT note_id, timestamp, author, content FROM case_notes WHERE case_id = ? ORDER BY note_id ASC",
                (case_id,)
            )
            case_data["notes"] = [dict(n) for n in cur_notes.fetchall()]

            # Fetch audit history
            cur_hist = conn.execute(
                "SELECT history_id, timestamp, action, details, author FROM case_history WHERE case_id = ? ORDER BY history_id ASC",
                (case_id,)
            )
            case_data["history"] = [dict(h) for h in cur_hist.fetchall()]

        return case_data

    def list_cases(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Lists cases with optional status, priority, and text search filters."""
        query = "SELECT * FROM cases WHERE 1=1"
        params: List[Any] = []

        if status and status.upper() != "ALL":
            query += " AND status = ?"
            params.append(status.upper())

        if priority and priority.upper() != "ALL":
            query += " AND priority = ?"
            params.append(priority.upper())

        if search_query:
            query += " AND (case_id LIKE ? OR title LIKE ? OR summary LIKE ?)"
            term = f"%{search_query}%"
            params.extend([term, term, term])

        query += " ORDER BY updated_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                c = dict(r)
                c["linked_transactions"] = json.loads(c.get("linked_transactions", "[]"))
                results.append(c)

        return results

    def update_case_status(
        self,
        case_id: str,
        new_status: str,
        author: str = "Compliance-Analyst",
        reason: str = ""
    ) -> bool:
        """Updates case status and appends entry to audit trail."""
        new_status = new_status.upper()
        if new_status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of: {self.VALID_STATUSES}")

        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT status FROM cases WHERE case_id = ?", (case_id,))
            row = cursor.fetchone()
            if row is None:
                return False

            old_status = row[0]
            if old_status == new_status:
                return True

            conn.execute(
                "UPDATE cases SET status = ?, updated_at = ? WHERE case_id = ?",
                (new_status, now, case_id)
            )

            detail_msg = f"Status changed from {old_status} to {new_status}."
            if reason:
                detail_msg += f" Reason: {reason}"

            conn.execute(
                """
                INSERT INTO case_history (case_id, timestamp, action, details, author)
                VALUES (?, ?, 'STATUS_CHANGE', ?, ?)
                """,
                (case_id, now, detail_msg, author)
            )
            conn.commit()

        logger.info(f"Updated case {case_id} status: {old_status} -> {new_status}")
        return True

    def add_case_note(self, case_id: str, author: str, content: str) -> int:
        """Appends a new investigation note and updates case modified timestamp."""
        if not content.strip():
            raise ValueError("Note content cannot be empty.")

        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT case_id FROM cases WHERE case_id = ?", (case_id,))
            if cursor.fetchone() is None:
                raise KeyError(f"Case '{case_id}' does not exist.")

            cur_note = conn.execute(
                """
                INSERT INTO case_notes (case_id, timestamp, author, content)
                VALUES (?, ?, ?, ?)
                """,
                (case_id, now, author, content.strip())
            )
            note_id = cur_note.lastrowid

            conn.execute("UPDATE cases SET updated_at = ? WHERE case_id = ?", (now, case_id))

            # Audit history record
            conn.execute(
                """
                INSERT INTO case_history (case_id, timestamp, action, details, author)
                VALUES (?, ?, 'NOTE_ADDED', ?, ?)
                """,
                (case_id, now, f"Note #{note_id} added by {author}.", author)
            )
            conn.commit()

        return note_id

    def link_transaction_to_case(
        self,
        case_id: str,
        transaction_id: str,
        author: str = "Compliance-Analyst"
    ) -> bool:
        """Associates a transaction reference to an ongoing case."""
        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT linked_transactions FROM cases WHERE case_id = ?", (case_id,))
            row = cursor.fetchone()
            if row is None:
                return False

            txns = json.loads(row[0])
            if transaction_id in txns:
                return True

            txns.append(transaction_id)
            conn.execute(
                "UPDATE cases SET linked_transactions = ?, updated_at = ? WHERE case_id = ?",
                (json.dumps(txns), now, case_id)
            )

            conn.execute(
                """
                INSERT INTO case_history (case_id, timestamp, action, details, author)
                VALUES (?, ?, 'TRANSACTION_LINKED', ?, ?)
                """,
                (case_id, now, f"Linked transaction reference: {transaction_id}", author)
            )
            conn.commit()

        return True

    def find_cases_by_transaction(self, transaction_id: str) -> List[Dict[str, Any]]:
        """Finds all cases that reference a specific transaction."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM cases WHERE linked_transactions LIKE ? ORDER BY updated_at DESC",
                (f"%{transaction_id}%",)
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                c = dict(r)
                txns = json.loads(c.get("linked_transactions", "[]"))
                if transaction_id in txns:
                    c["linked_transactions"] = txns
                    results.append(c)

        return results

    def get_case_metrics(self) -> Dict[str, Any]:
        """Calculates operational metrics across all investigation cases."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            open_count = conn.execute("SELECT COUNT(*) FROM cases WHERE status = 'OPEN'").fetchone()[0]
            investigating = conn.execute("SELECT COUNT(*) FROM cases WHERE status = 'INVESTIGATING'").fetchone()[0]
            escalated = conn.execute("SELECT COUNT(*) FROM cases WHERE status = 'ESCALATED_AML'").fetchone()[0]
            closed = conn.execute(
                "SELECT COUNT(*) FROM cases WHERE status IN ('CLOSED_FALSE_POSITIVE', 'CLOSED_CONFIRMED_ANOMALY')"
            ).fetchone()[0]
            critical = conn.execute("SELECT COUNT(*) FROM cases WHERE priority = 'CRITICAL'").fetchone()[0]

        return {
            "total_cases": total,
            "open_cases": open_count,
            "in_investigation": investigating,
            "escalated_aml": escalated,
            "closed_cases": closed,
            "critical_priority": critical
        }

    # =========================================================
    # ALERT CENTER PERSISTENCE & ACKNOWLEDGEMENT METHODS
    # =========================================================

    def save_or_update_alert(self, alert_data: Dict[str, Any]) -> str:
        """Saves or updates an AnomalyAlert in SQLite database."""
        alert_id = alert_data.get("alert_id")
        if not alert_id:
            raise ValueError("alert_data must contain an 'alert_id'")

        now = self._now_iso()
        supp_info = alert_data.get("supporting_info", {})
        supp_info_json = json.dumps(supp_info) if isinstance(supp_info, dict) else str(supp_info)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, transaction_id, anomaly_score, applied_threshold,
                    prediction, severity, amount, timestamp, model_id,
                    status, acknowledged, acknowledged_by, acknowledged_at,
                    linked_case_id, supporting_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(alert_id) DO UPDATE SET
                    anomaly_score = excluded.anomaly_score,
                    applied_threshold = excluded.applied_threshold,
                    prediction = excluded.prediction,
                    severity = excluded.severity,
                    supporting_info = excluded.supporting_info
                """,
                (
                    alert_id,
                    alert_data.get("transaction_id", "UNKNOWN"),
                    float(alert_data.get("anomaly_score", 0.0)),
                    float(alert_data.get("applied_threshold", 0.65)),
                    alert_data.get("prediction", "ANOMALY"),
                    alert_data.get("severity", "HIGH"),
                    float(alert_data.get("amount", 0.0)),
                    alert_data.get("timestamp", now),
                    alert_data.get("model_id", "SurveillanceEngine"),
                    alert_data.get("investigation_status", alert_data.get("status", "NEW")),
                    1 if alert_data.get("acknowledged", False) else 0,
                    alert_data.get("acknowledged_by"),
                    alert_data.get("acknowledged_at"),
                    alert_data.get("linked_case_id"),
                    supp_info_json
                )
            )
            conn.commit()
        return alert_id

    def save_alerts_batch(self, alerts_list: List[Dict[str, Any]]) -> int:
        """Batch inserts or updates a list of alerts."""
        count = 0
        for a in alerts_list:
            try:
                self.save_or_update_alert(a)
                count += 1
            except Exception as e:
                logger.warning(f"Error saving alert {a.get('alert_id')}: {e}")
        return count

    def list_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        search_query: Optional[str] = None,
        limit: int = 300
    ) -> List[Dict[str, Any]]:
        """Lists persisted alerts with filtering options."""
        query = "SELECT * FROM alerts WHERE 1=1"
        params: List[Any] = []

        if status and status.upper() != "ALL":
            query += " AND status = ?"
            params.append(status.upper())

        if severity and severity.upper() != "ALL":
            query += " AND severity = ?"
            params.append(severity.upper())

        if acknowledged is not None:
            query += " AND acknowledged = ?"
            params.append(1 if acknowledged else 0)

        if search_query:
            query += " AND (alert_id LIKE ? OR transaction_id LIKE ?)"
            term = f"%{search_query}%"
            params.extend([term, term])

        query += " ORDER BY anomaly_score DESC, timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                try:
                    item["supporting_info"] = json.loads(item.get("supporting_info", "{}"))
                except Exception:
                    item["supporting_info"] = {}
                item["acknowledged"] = bool(item.get("acknowledged", 0))
                results.append(item)
            return results

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single alert by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            item = dict(row)
            try:
                item["supporting_info"] = json.loads(item.get("supporting_info", "{}"))
            except Exception:
                item["supporting_info"] = {}
            item["acknowledged"] = bool(item.get("acknowledged", 0))
            return item

    def acknowledge_alert(self, alert_id: str, analyst_name: str = "Compliance-Analyst") -> bool:
        """Marks an alert as acknowledged by an analyst."""
        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE alerts
                SET acknowledged = 1,
                    acknowledged_by = ?,
                    acknowledged_at = ?,
                    status = CASE WHEN status = 'NEW' THEN 'ACKNOWLEDGED' ELSE status END
                WHERE alert_id = ?
                """,
                (analyst_name, now, alert_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def link_alert_to_case(self, alert_id: str, case_id: str) -> bool:
        """Links an alert to an investigation case and links transaction to the case."""
        alert = self.get_alert(alert_id)
        if not alert:
            return False

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE alerts
                SET linked_case_id = ?,
                    status = 'UNDER_INVESTIGATION'
                WHERE alert_id = ?
                """,
                (case_id, alert_id)
            )
            conn.commit()

        # Link transaction ID to the case
        tx_id = alert.get("transaction_id")
        if tx_id:
            self.link_transaction_to_case(case_id, tx_id, author=alert.get("acknowledged_by") or "Compliance-Analyst")

        return True

    def dismiss_alert(self, alert_id: str, analyst_name: str = "Compliance-Analyst", reason: str = "") -> bool:
        """Dismisses an alert as false alarm or benign anomaly."""
        now = self._now_iso()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE alerts
                SET status = 'DISMISSED',
                    acknowledged = 1,
                    acknowledged_by = ?,
                    acknowledged_at = ?
                WHERE alert_id = ?
                """,
                (analyst_name, now, alert_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_alert_metrics(self) -> Dict[str, Any]:
        """Returns summary counts across stored alerts."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            unack = conn.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0").fetchone()[0]
            crit = conn.execute("SELECT COUNT(*) FROM alerts WHERE severity = 'CRITICAL'").fetchone()[0]
            linked = conn.execute("SELECT COUNT(*) FROM alerts WHERE linked_case_id IS NOT NULL").fetchone()[0]
            new_alerts = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'NEW'").fetchone()[0]

        return {
            "total_alerts": total,
            "unacknowledged_alerts": unack,
            "critical_alerts": crit,
            "linked_to_case": linked,
            "new_alerts": new_alerts
        }

    # =========================================================
    # USER CONFIGURABLE SETTINGS PERSISTENCE
    # =========================================================

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value from SQLite settings table."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row is None:
                return default
            try:
                return json.loads(row[0])
            except Exception:
                return row[0]

    def set_setting(self, key: str, value: Any) -> None:
        """Stores a configuration value in SQLite settings table."""
        now = self._now_iso()
        val_str = json.dumps(value) if not isinstance(value, str) else value
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, val_str, now)
            )
            conn.commit()

