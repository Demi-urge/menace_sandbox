CREATE TABLE IF NOT EXISTS code_errors(source_menace_id TEXT, code_id INTEGER REFERENCES code(id), error_id INTEGER)
