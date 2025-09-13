import sqlite3
import os

DB_PATH = os.path.join("database", "voice_notes.db")

def add_column_if_missing(conn, table_name, column_def):
    column_name = column_def.split()[0]
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in c.fetchall()]
    if column_name not in columns:
        print(f"Adding column '{column_name}' to '{table_name}'")
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_def}")
        conn.commit()
    else:
        print(f"Column '{column_name}' already exists in '{table_name}'")

def update_null_timestamps(conn, table_name):
    c = conn.cursor()
    c.execute(f"UPDATE {table_name} SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
    conn.commit()

def main():
    if not os.path.exists(DB_PATH):
        print("Database not found. Creating new database...")
        conn = sqlite3.connect(DB_PATH)
    else:
        conn = sqlite3.connect(DB_PATH)

    # Fix notes table
    add_column_if_missing(conn, "notes", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    update_null_timestamps(conn, "notes")

    # Fix summaries table
    add_column_if_missing(conn, "summaries", "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    update_null_timestamps(conn, "summaries")

    conn.close()
    print("✅ Database fix completed successfully!")

if __name__ == "__main__":
    main()
