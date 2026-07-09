# Method

Method code is split into two groups:

- `backbones/`: VLM backbone-specific interfaces and compact result JSONs.
- `elicitation/`: preference elicitation implementations shared across backbones.

Copied backbone folders:

- `Qwen3vl-4B`
- `Qwen3vl-8B`
- `Qwen3vl-30B`
- `KimiK2.5`
- `Gemini-3-flash`

Copied elicitation folders:

- `Single-agent-Budgeted`
- `Single-agentLLM`
- `Single-agentLLM_score`
- `Single-agent_pair`
- `Valuator`
- `Valuator_judge`

Runtime logs, locks, process IDs, cache folders, and temporary folders were excluded.
