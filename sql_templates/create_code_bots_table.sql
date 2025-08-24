CREATE TABLE IF NOT EXISTS code_bots(source_menace_id TEXT, code_id INTEGER REFERENCES code(id), bot_id INTEGER)
