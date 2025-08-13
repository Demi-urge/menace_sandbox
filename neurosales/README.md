# Neurosales Memory System

In a previous discussion:

**User asked**
> Is this implemented?
>
> 1) Queues and stacks: The queue will maintain the last 3–6 messages max, focusing on recent triggers and objections. Stack memory will track CTA micro-chains (e.g., message > reply > escalation). Memory will be non-personal, with time-decay removing old context quickly. Used for managing tone shifts and pattern reversals (e.g., changing emotional approach when resistance is detected).

**Agent answered**

The repository implements a memory system with a bounded queue and a stack:

- **Queue** – `MemoryQueue` stores only 3–6 recent messages with optional trigger/objection metadata and prunes entries beyond a decay window, ensuring short-term, non-personal context.
- **Stack** – `MemoryStack` tracks “message → reply → escalation” CTA chains, also expiring old chains to enforce rapid context decay.
- **Combined usage** – `SalesConversationMemory` wraps both mechanisms, keeping message history and CTA steps with automatic pruning for time-decay.
- **Behavior tests** – unit tests verify queue bounds, CTA stack operations, and pruning logic.

