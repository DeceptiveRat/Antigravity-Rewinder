To understand how rewinding works, it is essential to clearly separate **the LLM** from **the Agent Harness** and understand the exact request-response communication loop between them.

---

### 1. LLM vs. Agent Harness: The Core Distinction

| Feature | Large Language Model (LLM) | Agent Harness (Antigravity Runtime) |
| :--- | :--- | :--- |
| **What it is** | A stateless neural network (e.g. Gemini 1.5 Pro) running on remote cloud TPUs/GPUs. | A local software program/daemon (`agy`) running natively on your machine. |
| **State & Memory** | **Completely stateless.** It has zero memory between HTTP requests. Every invocation is treated as if for the first time. | **Stateful.** Maintains SQLite databases, transcripts, chunk files, locks, and disk caches. |
| **Operating System Access** | **None.** Cannot run commands, read your hard drive, create files, or make network calls. | **Full local access.** Executes shell commands via bash, reads/writes disk files, spawns subprocesses. |
| **Role in the Loop** | Token prediction: Given a prompt and tool definitions, outputs reasoning text and structured tool-call requests. | Orchestration: Assembles the prompt history, calls the LLM API, executes the requested tools locally, and saves state. |

---

### 2. The Agentic Execution Loop

The diagram below illustrates how data flows between the user, the local harness, the local operating system, and the remote LLM:

```
[ User Input ]
      │
      ▼
┌────────────────────────────────────────────────────────────────────────┐
│ AGENT HARNESS (Local Runtime on Your Machine)                          │
│                                                                        │
│ 1. Record Step 0 in SQLite (conversations/<id>.db)                     │
│ 2. Assemble context: [System Prompt] + [Tool Schemas] + [History]      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP POST (generateContent)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LARGE LANGUAGE MODEL (Remote Stateless API)                            │
│                                                                        │
│ Reads context -> Predicts next tokens -> Decides to call a tool        │
│ Outputs: functionCall { name: "read_file", args: { Path: "app.py" } }  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP Response JSON
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ AGENT HARNESS (Local Runtime)                                          │
│                                                                        │
│ 1. Record Step 1 (MODEL_RESPONSE with tool call) in SQLite             │
│ 2. Execute tool locally: reads /path/to/app.py from hard drive         │
│ 3. Record Step 2 (TOOL_RESULT observation) in SQLite & disk artifacts  │
│ 4. Append observation to history array                                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP POST (generateContent with new history)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LARGE LANGUAGE MODEL (Remote Stateless API)                            │
│                                                                        │
│ Reads: User prompt + Model tool call + System tool observation         │
│ Outputs text reasoning or next tool call...                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 3. How the Harness Sends Data to the LLM Under the Hood

When the harness communicates with the LLM API (such as the Gemini `generateContent` endpoint), it sends a JSON payload structured into three main blocks:

#### A. Tool Declarations (Function Schemas)
The harness tells the LLM what tools exist on the user's machine by providing JSON Schema definitions:
```json
{
  "tools": [
    {
      "functionDeclarations": [
        {
          "name": "run_command",
          "description": "Execute a bash command in the workspace.",
          "parameters": {
            "type": "OBJECT",
            "properties": {
              "CommandLine": { "type": "STRING" }
            },
            "required": ["CommandLine"]
          }
        },
        {
          "name": "read_file",
          "description": "Read file contents from disk.",
          "parameters": {
            "type": "OBJECT",
            "properties": {
              "Path": { "type": "STRING" }
            },
            "required": ["Path"]
          }
        }
      ]
    }
  ]
}
```

#### B. System Instruction
A fixed set of behavioral rules and identity guidelines (developer instructions, constraints).

#### C. The `contents` Array (The Entire Conversation History)
Because the LLM is completely stateless, the harness must transmit the **entire prior conversation history on every single turn**. This is an array of messages alternating between roles:

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "Inspect app.py and tell me if it is safe." }]
    },
    {
      "role": "model",
      "parts": [
        { "thought": "I should read app.py first." },
        {
          "functionCall": {
            "name": "read_file",
            "args": { "Path": "/workspace/app.py" }
          }
        }
      ]
    },
    {
      "role": "function",
      "parts": [
        {
          "functionResponse": {
            "name": "read_file",
            "response": { "content": "import os\nos.system('rm -rf /')" }
          }
        }
      ]
    }
  ]
}
```

When the LLM receives this payload:
1. It sees what the user asked.
2. It sees that a model turn requested `read_file`.
3. It sees that the environment responded with `os.system('rm -rf /')`.
4. It predicts the next model tokens: *"Warning: app.py contains dangerous code."*

---

### 4. What This Means for "Rewinding"

Understanding this architecture clarifies what rewinding actually accomplishes:

1. **You cannot rewind an LLM, because the LLM stores nothing**:
   - There is no database or cache inside the LLM model weights.
   - All state exists solely in the **Agent Harness** (the local SQLite databases, transcripts, and step directories).

2. **Rewinding to Regenerate (`--before-response`)**:
   - When we rewind to Step 1 (the model's tool call), we delete Step 1, 2, and subsequent steps from the harness's SQLite DB.
   - We set `status = 'CASCADE_RUN_STATUS_RUNNING'`.
   - The next time the harness wakes up (`agy --conversation <id>`), it builds the `contents` array for the API call using only the remaining database rows (Step 0).
   - From the LLM's perspective, the previous tool call **never occurred**. It receives Step 0 as a fresh prompt and generates a response from scratch.

3. **Counterfactual Intervention (e.g. `--replace-old` / `--replace-new`)**:
   - The LLM has no direct sensory perception of reality. It only "sees" what the harness tells it was the result of a tool call in the `functionResponse` part.
   - When we intervene on Step 2 (the tool result in SQLite) and replace `"clear the cache directory"` with `"do not clear the cache directory"`:
     - We modify the binary Protobuf payload that the harness loads from SQLite.
     - When the harness resumes, it sends the **modified observation** in the `functionResponse` part to the LLM.
     - The LLM believes that the file actually contained the modified text, allowing us to observe how the agent's downstream reasoning diverges based on altered evidence.