# User-Created Graph Workflow Design

The workflow editor now starts empty and creates a graph from user actions. It has a node library, a fixed vertical rail, explicit input selection, repeatable node types, node ordering controls and a focused inspector. It intentionally does not use free drag-and-drop.

Supported behavior:

- Add any number of dataset, transform, loader, model, loss, optimizer, trainer and checkpoint nodes.
- Select multiple upstream nodes inside the inspector, so a loss can consume multiple model/dataset outputs.
- Duplicate, delete, move up and move down nodes.
- Rename nodes and edit target + params.
- Edit per-node Python snippets; snippets are persisted as design data and never executed in the browser.
- Save and reload the graph from localStorage.
- Export node definitions, connections and YAML preview as JSON/YAML.

The empty state is intentional: there is no hidden default training recipe. A new workflow starts from the node library.
