import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time


@dataclass(frozen=True)
class HistoryEntry:
    session_id: int
    segment: int
    recorded_at: float
    source: str
    source_final: bool
    translation: str
    translation_final: bool
    speaker: str = ""

    @property
    def key(self):
        return self.session_id, self.segment

    @property
    def order_key(self):
        return self.session_id, self.recorded_at, self.segment

    @property
    def heading(self):
        heading = f"{datetime.fromtimestamp(self.recorded_at):%Y-%m-%d %H:%M:%S}"
        if self.speaker:
            heading += f" · {SPEAKER_LABELS.get(self.speaker, self.speaker)}"
        if (self.source and not self.source_final) or (self.translation and not self.translation_final):
            heading += " · 草稿"
        return heading


SPEAKER_LABELS = {"A": "A（系统播放）", "B": "B（麦克风）"}
ENTRY_COLUMNS = "session_id, segment, recorded_at, source, source_final, translation, translation_final, speaker"


class ConversationHistory:
    def __init__(self, path):
        self.path = path
        self.documents_dir = path.parent / "对话记录"
        self._db = None
        self.start_session()

    def _connect(self):
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, timeout=0.05)
            try:
                db.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS settings (
                        name TEXT PRIMARY KEY,
                        value INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY
                    );
                    CREATE TABLE IF NOT EXISTS entries (
                        session_id INTEGER NOT NULL,
                        segment INTEGER NOT NULL,
                        recorded_at REAL NOT NULL,
                        source TEXT NOT NULL DEFAULT '',
                        source_final INTEGER NOT NULL DEFAULT 0,
                        translation TEXT NOT NULL DEFAULT '',
                        translation_final INTEGER NOT NULL DEFAULT 0,
                        speaker TEXT NOT NULL DEFAULT '',
                        PRIMARY KEY (session_id, segment)
                    );
                    CREATE INDEX IF NOT EXISTS entries_by_time ON entries (session_id, recorded_at, segment);
                    CREATE TABLE IF NOT EXISTS documents (
                        session_id INTEGER PRIMARY KEY,
                        filename TEXT NOT NULL UNIQUE,
                        needs_write INTEGER NOT NULL DEFAULT 1
                    );
                """)
                if "speaker" not in {column[1] for column in db.execute("PRAGMA table_info(entries)")}:
                    db.execute("ALTER TABLE entries ADD COLUMN speaker TEXT NOT NULL DEFAULT ''")
                db.commit()
            except sqlite3.Error:
                db.close()
                raise
            self._db = db
        return self._db

    def enabled(self):
        if not self.path.exists():
            return False
        row = self._connect().execute("SELECT value FROM settings WHERE name = 'record_history'").fetchone()
        return bool(row and row[0])

    def set_enabled(self, enabled):
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES ('record_history', ?)", (enabled,))

    def start_session(self):
        self._session_id = None
        self._started_at = datetime.now()

    def prepare_document(self):
        if self._session_id is None:
            with self._connect() as db:
                session_id = db.execute("INSERT INTO sessions DEFAULT VALUES").lastrowid
                filename = f"{self._started_at:%Y-%m-%d_%H-%M-%S}_{session_id:06d}.txt"
                db.execute("INSERT INTO documents (session_id, filename) VALUES (?, ?)", (session_id, filename))
            self._session_id = session_id

    def record(self, updates):
        self.prepare_document()
        session_id = self._session_id
        changed = {}
        with self._connect() as db:
            for event, is_source in updates:
                column = "source" if is_source else "translation"
                row = db.execute(f"""
                    INSERT INTO entries (session_id, segment, recorded_at, {column}, {column}_final, speaker)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, segment) DO UPDATE SET
                        {column} = excluded.{column}, {column}_final = excluded.{column}_final
                    WHERE excluded.{column}_final >= entries.{column}_final
                    RETURNING {ENTRY_COLUMNS}
                """, (session_id, event.segment, event.recorded_at or time(), event.text, event.final,
                      event.speaker)).fetchone()
                if row is not None:
                    entry = HistoryEntry(*row)
                    changed[entry.key] = entry
            if changed:
                db.execute("UPDATE documents SET needs_write = 1 WHERE session_id = ?", (session_id,))
        return changed.values()

    def recent(self, limit, *, before=None):
        if not self.path.exists():
            return []
        where = "WHERE (session_id, recorded_at, segment) < (?, ?, ?)" if before is not None else ""
        rows = self._connect().execute(f"""
            SELECT {ENTRY_COLUMNS} FROM entries {where}
            ORDER BY session_id DESC, recorded_at DESC, segment DESC LIMIT ?
        """, (*(before or ()), limit))
        return [HistoryEntry(*row) for row in rows]

    def sync_documents(self):
        if not self.path.exists():
            return
        db = self._connect()
        pending = db.execute("SELECT session_id FROM documents WHERE needs_write = 1").fetchall()
        failure = None
        for (session_id,) in pending:
            try:
                with db:
                    # Keep the document snapshot and its write flag in sync across app instances.
                    db.execute("BEGIN IMMEDIATE")
                    row = db.execute(
                        "SELECT filename FROM documents WHERE session_id = ? AND needs_write = 1", (session_id,),
                    ).fetchone()
                    if row is None:
                        continue
                    with closing(db.execute(f"""
                        SELECT {ENTRY_COLUMNS} FROM entries WHERE session_id = ? ORDER BY recorded_at, segment
                    """, (session_id,))) as entries:
                        self._write_document(row[0], entries)
                    db.execute("UPDATE documents SET needs_write = 0 WHERE session_id = ?", (session_id,))
            except OSError as exc:
                failure = exc
        if failure is not None:
            raise failure

    def _write_document(self, filename, rows):
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8-sig", dir=self.documents_dir, delete=False) as document:
                temporary_path = Path(document.name)
                document.write("AITranslate 对话记录\n\n")
                for row in rows:
                    entry = HistoryEntry(*row)
                    document.write(f"{entry.heading}\n")
                    if entry.source:
                        document.write(f"原文\n{entry.source}\n")
                    if entry.translation:
                        document.write(f"译文\n{entry.translation}\n")
                    document.write("\n")
            temporary_path.replace(self.documents_dir / filename)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
