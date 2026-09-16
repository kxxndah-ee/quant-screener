"""
Database models and connection management for the Quant Screener application.
Uses local SQLite database (data/app.db).
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")
CACHE_DIR = os.path.join(DATA_DIR, "cache")


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with row factory enabled."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes all database tables and populates default watchlist if empty."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Watchlist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                market TEXT,
                added_at TEXT NOT NULL,
                notes TEXT,
                group_name TEXT DEFAULT '기본그룹'
            )
        """)

        # Migration: Ensure group_name column exists
        cursor.execute("PRAGMA table_info(watchlist)")
        wl_cols = [r["name"] for r in cursor.fetchall()]
        if "group_name" not in wl_cols:
            cursor.execute("ALTER TABLE watchlist ADD COLUMN group_name TEXT DEFAULT '기본그룹'")

        # 2. Universe (Active & Delisted stocks for survivorship-bias-free analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universe (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                market TEXT,
                sector TEXT,
                industry TEXT,
                is_active INTEGER DEFAULT 1,
                delist_date TEXT,
                last_updated TEXT
            )
        """)

        # 3. Predictions Diary
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                code TEXT NOT NULL,
                name TEXT NOT NULL,
                score REAL NOT NULL,
                label TEXT NOT NULL,
                rsi REAL,
                ma_align INTEGER,
                vol_surge REAL,
                bb_pct REAL,
                target_tp REAL,
                target_sl REAL,
                status TEXT DEFAULT 'PENDING',
                created_at TEXT NOT NULL
            )
        """)

        # 4. Settlements (Forward testing auto-settlement results)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settlements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER UNIQUE,
                code TEXT NOT NULL,
                target_date TEXT NOT NULL,
                actual_open REAL,
                actual_high REAL,
                actual_low REAL,
                actual_close REAL,
                actual_return REAL,
                hit_status TEXT,
                exit_type TEXT,
                settled_at TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)

        # 5. Trade Executions (Actual/Simulated trades)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_date TEXT NOT NULL,
                code TEXT NOT NULL,
                name TEXT NOT NULL,
                entry_price REAL NOT NULL,
                shares INTEGER NOT NULL,
                exit_date TEXT,
                exit_price REAL,
                pnl_amount REAL,
                return_pct REAL,
                exit_reason TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # 6. Portfolio Risk State
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_state (
                id INTEGER PRIMARY KEY,
                date TEXT NOT NULL,
                current_equity REAL DEFAULT 10000000.0,
                daily_realized_loss REAL DEFAULT 0.0,
                is_circuit_breaker_active INTEGER DEFAULT 0,
                consecutive_losses INTEGER DEFAULT 0,
                cooldown_active INTEGER DEFAULT 0
            )
        """)

        # Populate initial risk state if empty
        cursor.execute("SELECT COUNT(*) FROM risk_state")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO risk_state (id, date, current_equity, daily_realized_loss, is_circuit_breaker_active, consecutive_losses, cooldown_active)
                VALUES (1, ?, 10000000.0, 0.0, 0, 0, 0)
            """, (datetime.now().strftime("%Y-%m-%d"),))

        # Populate default watchlist if empty
        cursor.execute("SELECT COUNT(*) FROM watchlist")
        if cursor.fetchone()[0] == 0:
            defaults = [
                ("005930", "삼성전자", "KOSPI", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "반도체 대장주"),
                ("000660", "SK하이닉스", "KOSPI", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "HBM/메모리"),
                ("035420", "NAVER", "KOSPI", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "플랫폼"),
                ("005380", "현대차", "KOSPI", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "완성차/모빌리티"),
                ("068270", "셀트리온", "KOSPI", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "바이오시밀러"),
            ]
            cursor.executemany("""
                INSERT INTO watchlist (code, name, market, added_at, notes)
                VALUES (?, ?, ?, ?, ?)
            """, defaults)

        conn.commit()


def update_watchlist_group(code: str, group_name: str, db_path: str = DB_PATH) -> bool:
    """Updates group_name for a given stock in watchlist."""
    clean_grp = group_name.strip() if group_name and group_name.strip() else "기본그룹"
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE watchlist SET group_name = ? WHERE code = ?", (clean_grp, code))
        conn.commit()
        return cursor.rowcount > 0


def batch_update_watchlist_groups(codes: List[str], group_name: str, db_path: str = DB_PATH) -> int:
    """Batch updates group_name for multiple stocks in watchlist."""
    if not codes:
        return 0
    clean_grp = group_name.strip() if group_name and group_name.strip() else "기본그룹"
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(codes))
        cursor.execute(f"UPDATE watchlist SET group_name = ? WHERE code IN ({placeholders})", [clean_grp] + codes)
        conn.commit()
        return cursor.rowcount


def get_watchlist_groups(db_path: str = DB_PATH) -> List[str]:
    """Returns sorted list of distinct watchlist group names."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT COALESCE(NULLIF(group_name, ''), '기본그룹') AS grp FROM watchlist ORDER BY grp ASC")
        rows = cursor.fetchall()
        groups = [r["grp"] for r in rows]
        if not groups:
            return ["기본그룹"]
        if "기본그룹" in groups:
            groups.remove("기본그룹")
            groups.insert(0, "기본그룹")
        return groups


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
