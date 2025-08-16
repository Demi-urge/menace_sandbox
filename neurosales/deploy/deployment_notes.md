# Deployment notes

The persistent memory modules introduce several new SQL tables:

- `embedding_messages` for embedding and vector memories
- `reaction_pairs` for reaction history
- `preference_messages` for the preference engine
- `emotion_entries` for emotion tracking
- `reward_entries` for the reward ledger
- `rl_policy_weights` and `replay_experiences` for the RL ranking system

Run `create_session()` or your migration tool before deploying to create these
tables.

When building the Docker image with `HEAVY_DEPS=true` the build now runs
`scripts/setup_heavy_deps.py --download-only` so the MiniLM embedding weights are
prefetched into the image.

## Persistent volume

Create a PersistentVolumeClaim for the database:

```bash
kubectl apply -f - <<'YAML'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: neurosales-data
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
YAML
```

Mount the claim by enabling persistence in the Helm values.
Set `NEURO_DB_URL` to use this path, e.g. `sqlite:////data/neuro.db`.
For Postgres deployments configure `NEURO_DB_URL` with your connection string.
Set `NEURO_DB_POOL_SIZE` and `NEURO_DB_MAX_OVERFLOW` to control SQLAlchemy
connection pooling when not passed directly to `create_session()`.
`DatabaseConnectionManager` also reads `NEURO_POSTGRES_URLS`, `NEURO_MONGO_URLS`
and `NEURO_ENABLE_DB_LOAD_BALANCING` when these options are not supplied in
code. Each URL list is comma separated.
Set `NEURO_REDIS_URL` to point to a Redis instance and
`NEURO_MEMCACHED_SERVERS` for Memcached hosts if you wish to use them.
Configure the remaining external services with:
`NEURO_POSTGRES_URLS`, `NEURO_MONGO_URLS`, `NEURO_ENABLE_DB_LOAD_BALANCING`,
`NEURO_OPENAI_KEY`,
`NEURO_PINECONE_INDEX`, `NEURO_PINECONE_KEY`, `NEURO_PINECONE_ENV`,
`NEURO_NEO4J_URI`, `NEURO_NEO4J_USER`, `NEURO_NEO4J_PASS` and
`NEURO_PROXY_LIST`.

`NEURO_PROXY_LIST` accepts a comma-separated list of proxy URLs such as
`http://user:pass@host:3128`. `APIScraper` and `DynamicHarvester` will randomly
select one of these proxies for each request, providing rudimentary rotation.
Make sure the URLs include the `http` scheme and valid credentials. A mix of
unreachable or misconfigured proxies can lead to intermittent timeouts when the
harvester is running in the sandbox.

For local development see [deploy/docker-compose.yml](docker-compose.yml),
which provisions Postgres with
`postgresql://neuro:neuro@db/neuro`. Add the same value for
`NEURO_DB_URL` in your `.env` so compose deployments connect to the bundled
database service.
Scripts such as `migrate.py`, `backup_db.py` and `restore_db.py` as well as the API gateway automatically load this `.env` file using `python-dotenv`.

## Database persistence

Set `NEURO_DB_URL` in your `.env` to an SQLite path or a PostgreSQL URL.
After exporting this variable run:

```bash
python scripts/migrate.py upgrade head
```

This creates or updates all tables. You can call `run_migrations()` from your
startup scripts (the API gateway already does) so migrations apply
automatically. Run the command again whenever the models change or new
migration files are added.

Create backups periodically using:

```bash
python scripts/backup_db.py
```

Restore the most recent backup with:

```bash
python scripts/restore_db.py backup.json
```
These helper scripts read environment variables from `.env` automatically.
`restore_db.py` calls `psql` automatically when the file contains a `pg_dump` dump.

### Database environment variables

`NEURO_DB_POOL_SIZE` and `NEURO_DB_MAX_OVERFLOW` tune the SQLAlchemy connection
pool when explicit `pool_size` or `max_overflow` values are not passed to
`create_session()`. Example values:

```bash
export NEURO_DB_POOL_SIZE=5
export NEURO_DB_MAX_OVERFLOW=10
```

Provide comma-separated lists via `NEURO_POSTGRES_URLS` or `NEURO_MONGO_URLS`
when running multiple replicas and set `NEURO_ENABLE_DB_LOAD_BALANCING=true` to
distribute traffic by hashing the shard key:

```bash
export NEURO_POSTGRES_URLS=postgresql://db1,postgresql://db2
export NEURO_MONGO_URLS=mongodb://m1:27017,mongodb://m2:27017
export NEURO_ENABLE_DB_LOAD_BALANCING=true
```

After exporting these variables you can instantiate the manager without
arguments and connections will be pooled and balanced automatically:

```python
from neurosales.db_manager import DatabaseConnectionManager

mgr = DatabaseConnectionManager()
pg = mgr.get_postgres("user-id")
```

### DatabaseConnectionManager

Use `DatabaseConnectionManager` when your deployment spans multiple database
instances. Provide a list of `postgres_urls` or `mongo_urls` and enable
`enable_load_balancing` to distribute traffic:

```python
from neurosales.db_manager import DatabaseConnectionManager

mgr = DatabaseConnectionManager({
    "postgres_urls": ["postgresql://db1", "postgresql://db2"],
    "mongo_urls": ["mongodb://m1", "mongodb://m2"],
    "enable_load_balancing": True,
})
```

Consider tuning connection pooling through `create_session(pool_size=N)` or
running an external pooler like PgBouncer when scaling out.
Alternatively set `NEURO_DB_POOL_SIZE` and `NEURO_DB_MAX_OVERFLOW` so
`create_session()` applies these values automatically.

## API key secret

Create a secret containing the API key and reference it in the chart:

```bash
kubectl create secret generic neurosales-api-key \
  --from-literal=NEURO_API_KEY=<your-key>
```

## Ingress TLS

To secure traffic set `ingress.enabled=true` and supply a certificate:

```yaml
ingress:
  enabled: true
  host: my.domain
  tls:
    secretName: my-tls-secret
```

