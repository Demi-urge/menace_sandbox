CREATE VIRTUAL TABLE IF NOT EXISTS code_fts USING fts5(summary, code)
