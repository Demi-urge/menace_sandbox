# Embedding Backfill Watch Mode

`vector_service.embedding_backfill` can run in a daemon-like mode to keep
embeddings fresh. The `--watch` flag polls each database listed in
`vector_service/embedding_registry.json` and vectorises any new or updated
records using `SharedVectorService`.

## systemd example

Create a systemd unit to run the watcher continuously:

```ini
[Unit]
Description=Embedding backfill watcher
After=network.target

[Service]
# Adjust the path to the repository as needed
WorkingDirectory=/path/to/menace_sandbox
ExecStart=/usr/bin/python -m vector_service.embedding_backfill --watch
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable --now embedding-backfill.service
```

The process will monitor every registered database and invoke
`SharedVectorService.vectorise_and_store` whenever new content appears.
