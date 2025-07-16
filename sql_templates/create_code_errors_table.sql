CREATE TABLE IF NOT EXISTS code_errors(code_id INTEGER REFERENCES code(id), error_id INTEGER)
