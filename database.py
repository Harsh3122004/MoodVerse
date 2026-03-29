"""
MoodVerse — Database Setup
SQLite with users, watchlist, and review_history tables.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'moodflix.db')


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Watchlist table
    c.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            movie_id     INTEGER NOT NULL,
            title        TEXT NOT NULL,
            genres       TEXT,
            avg_rating   REAL,
            release_year INTEGER,
            poster       TEXT,
            added_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, movie_id)
        )
    ''')

    # Review history table (for sentiment analysis)
    c.execute('''
        CREATE TABLE IF NOT EXISTS review_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            review_text  TEXT NOT NULL,
            sentiment    TEXT NOT NULL,
            confidence   REAL NOT NULL,
            analyzed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Recommendation history
    c.execute('''
        CREATE TABLE IF NOT EXISTS rec_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            mood        TEXT,
            genres      TEXT,
            movie_ids   TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Sessions table for token persistence
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT PRIMARY KEY,
            user_id     INTEGER NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # ── PERSISTENT CACHE FOR POSTERS ──────────────────────────────────────
    c.execute('''
        CREATE TABLE IF NOT EXISTS poster_cache (
            key         TEXT PRIMARY KEY,
            poster_url  TEXT,
            plot        TEXT,
            rating      TEXT,
            year        TEXT,
            source      TEXT,
            success     INTEGER DEFAULT 1,
            cached_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ── PERFORMANCE INDEXES ───────────────────────────────────────────────
    c.execute('CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_history_user ON review_history(user_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_rec_user ON rec_history(user_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_session_user ON sessions(user_id)')

    conn.commit()
    conn.close()
    print('Database initialized at', DB_PATH)


if __name__ == '__main__':
    init_db()
