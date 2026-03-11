"""check_db.py — run before train_master.py to catch corrupted SQLite files."""
import os
import sqlite3

db = "database.sqlite"
if os.path.exists(db):
    try:
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA integrity_check")
        conn.close()
        print("DB OK")
    except sqlite3.DatabaseError:
        print("DB corrupted — deleting so train_master.py can recreate it.")
        os.remove(db)
else:
    print("No database.sqlite found — will be created by train_master.py.")
