SELECT cb.code_id, b.content_hash AS content_hash
FROM code_bots cb
JOIN code c ON cb.code_id = c.id
JOIN bots b ON cb.bot_id = b.id
WHERE cb.bot_id=?
