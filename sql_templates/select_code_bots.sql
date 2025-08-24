SELECT cb.code_id FROM code_bots cb JOIN code c ON cb.code_id = c.id WHERE cb.bot_id=? AND c.source_menace_id=?
