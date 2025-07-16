SELECT c.* FROM code AS c JOIN code_fts f ON f.rowid = c.id WHERE code_fts MATCH ?
