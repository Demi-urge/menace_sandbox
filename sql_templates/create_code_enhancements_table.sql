CREATE TABLE IF NOT EXISTS code_enhancements(source_menace_id TEXT, code_id INTEGER REFERENCES code(id), enhancement_id INTEGER)
