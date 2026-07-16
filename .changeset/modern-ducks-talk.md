---
'@mastra/core': patch
---

Fixed `ProcessorContext.agent` so processors receive the owning agent during standard and durable runs, including cross-process durable continuations. Processors can now access agent configuration such as memory from every processor hook.
