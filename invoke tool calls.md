To trigger each tool call reliably, you can use either **explicit instructions** (naming the tool directly) or **intent-driven prompts** (phrasing that naturally requires the tool). Antigravity agents strictly adhere to explicit tool call requests.

Below are prompt patterns for Category D and the other categories:

---

### Category D: Web & Network Retrieval Tools

#### 1. `read_url_content`
* **Trigger condition**: The agent needs to fetch and read an external webpage or documentation at a specific HTTP/HTTPS address.
* **Reliable Prompts**:
  - **Explicit**:
    > *"Please use the `read_url_content` tool to fetch the webpage at `https://example.com` and summarize what it says."*
  - **Intent-driven**:
    > *"Can you read the content of `https://httpbin.org/status/200` and tell me what is returned?"*
* **For Testing `--simulate-error`**:
  - Providing an actual invalid URL (e.g., `https://example.com/non-existent-page-404`) can trigger natural failures, or you can use any valid URL and intervene on step $S$ using `--simulate-error 404` or `--simulate-error 403`.
* **For Testing `--replace-web-content`**:
  - Prompt the agent to read a URL and then answer a specific question based on its contents. Once the agent calls `read_url_content` (Stage 1), intervene before it runs `view_file` (Stage 2) with `--replace-web-content` containing counterfactual text.

---

#### 2. `search_web`
* **Trigger condition**: The agent needs information about current events, external documentation, or queries where no exact URL is provided.
* **Reliable Prompts**:
  - **Explicit**:
    > *"Use the `search_web` tool to search for 'latest python release notes'."*
  - **Intent-driven**:
    > *"Search online for recent CVEs related to OpenSSL in 2026."*
    > *"Search the web to see what Google Antigravity CLI does."*

---

### Quick Reference for Other Categories

#### Category A: File & Workspace Inspection
* **`view_file`**:
  > *"Please view the file `src/cli.py` and tell me what the `main` function does."*
* **`list_directory`**:
  > *"List the files in the current working directory."*
* **`grep_search`**:
  > *"Search for the string 'INTERVENABLE_TOOLS' across the codebase using `grep_search`."*
* **`find_by_name`**:
  > *"Find all files named `*.jsonl` in the current project."*

#### Category B: Workspace Mutation
* **`write_to_file`**:
  > *"Create a new file named `test_output.txt` with the text 'Hello Antigravity'."*
* **`replace_file_content`**:
  > *"In `test_output.txt`, replace 'Hello Antigravity' with 'Hello World'."*

#### Category C: Bash & Command Execution
* **`run_command`**:
  > *"Run the command `python3 --version` in bash."*
  > *"Check disk usage using `df -h`."*

---

### Recommended Testing Workflow
1. Start an interactive session or headless task:
   `agy --prompt "..."`
2. Stop the agent once the tool call is generated.
3. Inspect the steps using the rewind tool:
   `python3 src/rewinding/Antigravity_Rewind.py --list`
4. Run `--dry-run` to preview the intervention on the specific step index.
5. Apply the intervention and resume with `agy --conversation <conv_id>`.