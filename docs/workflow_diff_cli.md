# Workflow Diff CLI

`tools/workflow_diff_cli.py` prints a unified diff between two workflow
specifications.

The command looks for workflow specification files saved by
[`workflow_spec.save_spec`](../workflow_spec.py) inside a `workflows`
subdirectory.  If the child workflow's metadata references a stored diff
(`diff_path`) for the supplied parent it is printed directly.  Otherwise the
diff is regenerated on the fly using Python's standard `difflib`.

## Usage

```bash
python -m tools.workflow_diff_cli <parent_id> <child_id>
python -m tools.workflow_diff_cli <parent_id> <child_id> --dir /path/to/run
```

The optional `--dir` flag points to the directory containing the `workflows`
folder.  If omitted it defaults to the current directory or the value of the
`WORKFLOW_OUTPUT_DIR` environment variable.
