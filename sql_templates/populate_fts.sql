INSERT OR IGNORE INTO code_fts(rowid, summary, code) SELECT id, summary, code FROM code
