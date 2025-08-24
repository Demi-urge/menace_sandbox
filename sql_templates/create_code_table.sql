CREATE TABLE IF NOT EXISTS code(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT,
    template_type TEXT,
    language TEXT,
    version TEXT,
    complexity REAL,
    summary TEXT,
    revision INTEGER DEFAULT 0,
    source_menace_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_code_source_menace_id ON code(source_menace_id);
