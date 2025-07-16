CREATE TABLE IF NOT EXISTS code(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT,
    template_type TEXT,
    language TEXT,
    version TEXT,
    complexity REAL,
    summary TEXT,
    revision INTEGER DEFAULT 0
)
