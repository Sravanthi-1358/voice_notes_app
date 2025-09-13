import sqlite3

conn = sqlite3.connect("database/voice_notes.db")
c = conn.cursor()

# Add 'created_at' column if missing
try:
    c.execute("ALTER TABLE summaries ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    print("✅ Column added successfully.")
except sqlite3.OperationalError:
    print("⚠️ Column already exists.")

# Update NULL timestamps
c.execute("UPDATE summaries SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
conn.commit()
conn.close()

print("✅ Database initialized.")
