# Mutation logging and lineage

`mutation_logger` records code changes along with their reason, trigger and performance impact. Each call writes to `evolution_history.db` and returns a row id so later mutations can reference their parents.

## Logging mutations

```python
from menace import mutation_logger as ml

root = ml.log_mutation("bootstrap", "initial", "manual", 1.0, workflow_id=1)
child = ml.log_mutation(
    "refactor", "experiment", "auto", 0.8, workflow_id=1, parent_id=root
)
```

## Querying lineage and interpreting the tree

`build_lineage` reconstructs nested dictionaries showing how each mutation descends from previous ones:

```python
tree = ml.build_lineage(1)
print(tree)
```

Example output:

```json
[
  {"rowid": 1, "action": "bootstrap", "children": [
    {"rowid": 2, "action": "refactor", "children": []}
  ]}
]
```

Here `refactor` (rowid 2) is a child of the `bootstrap` change (rowid 1), meaning the second mutation derives from the first. Nodes with multiple entries in `children` represent branches. When combined with ROI or error metrics they reveal which paths improve performance and which regress.

`MutationLineage.build_tree()` expands nodes with patch statistics such as ROI and complexity:

```python
from menace.mutation_lineage import MutationLineage

lineage = MutationLineage()
rich_tree = lineage.build_tree(1)
```

## Branching for A/B tests

`MutationLineage` can clone an existing patch into a new branch for experimentation:

```python
variant_id = lineage.clone_branch_for_ab_test(existing_patch_id, "tweak search strategy")
```

You can also backtrack from a failing patch to the last positive ROI ancestor:

```python
path = lineage.backtrack_failed_path(variant_id)
```

## CLI usage

The same helpers are available via a small command line interface:

```bash
python -m menace.mutation_lineage tree 1
python -m menace.mutation_lineage clone 42 --description "test variant"
```
