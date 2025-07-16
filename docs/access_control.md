# Database Access Control

`DatabaseRouter` enforces a simple role based model.

Three roles are available:

- `read` – may only execute read only queries.
- `write` – may perform reads and write operations.
- `admin` – includes write access and any future administrative actions.

The router is initialised with a mapping of bot name to role via the
`bot_roles` parameter. Each public method accepts an optional
`requesting_bot` argument which is checked against the stored role.
Unauthorized operations raise `PermissionError`.

All queries are logged to the `audit_log` table inside `MenaceDB` with the
bot identity, action and details so administrators can review database usage.
