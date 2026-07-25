# llama.cpp tool-call continuation via `return_prefill` -- research and test plan

## Symptom

When a generation is interrupted mid-thought (mid `reasoning_content`) and pi
auto-retries with `/continue`, the model's tool call is emitted as **raw text**
in `content` (e.g. `<function=read_file><parameter=path>...</function>` for
Qwen3, or `<|tool_calls_section_begin|>...` for gpt-oss) instead of a structured
`tool_calls` array, and `finish_reason` is `"stop"` instead of `"tool_calls"`.
pi therefore prints the markers as text and never executes the tool.

Server under test: `http://localhost:5021` (llama-server), model
`Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf`.

## How `/continue` + prefill works in pi

Code path (`/continue` and auto-retry-on-error share it):

1. `interactive-mode.ts` `handleContinueCommand()` calls `AgentSession.continue()`
   (`packages/coding-agent/src/core/agent-session.ts:1252`).
2. `AgentSession.continue()` calls `_continueOnce()` (`agent-session.ts:1279`).
3. `_continueOnce()` auto-detects a prefill candidate via
   `_detectPrefillCandidate()` (`agent-session.ts:2988`) which calls
   `_canPrefillContinue()` (`agent-session.ts:2967`).
4. `_canPrefillContinue()` returns `false` if the assistant message contains a
   `toolCall` block, so pi **never** attempts to prefill a message that already
   has a tool call. It only prefills reasoning/text.
5. The candidate is removed from the transcript (`_capturePrefill`,
   `agent-session.ts:3002`) and passed to `Agent.continue(prefill)`
   (`packages/agent/src/agent.ts:360`).
6. `Agent.continue()` rejects prefill with tool calls
   (`agent.ts:378`: `throw new Error("Cannot continue a message with a tool
   call")`) and non-openai-completions providers (`agent.ts:381`).
7. `runContinuation()` (`agent.ts:530`) sets `returnPrefill: true` in the loop
   config; `assertContinuableLastMessage()` (`agent-loop.ts:185`) enforces the
   same constraints on pi's side.
8. `openai-completions.ts:701` sets `return_prefill: true` on the request body.
9. During streaming, `Agent.checkPrefillEcho()` (`agent.ts:453`) verifies the
   echoed `reasoning_content` + `content` prefixes match the prefill; for a
   reasoning-only prefill the `content` prefix check is trivially satisfied
   (empty string), so the raw-text tool call passes verification.
10. Auto-retry-on-error drives the same path: `_handlePostAgentRun()`
    (`agent-session.ts:1322`) calls `_prepareRetry()` which returns `true`, and
    the `while` loop in `continue()` calls `_continueOnce()` again, re-capturing
    the partial as prefill.

So the failing real-world path is **case 3 below**: a reasoning-only prefill
(no tool call in the prefill message) that primes the model to emit a tool call.
pi does not (and cannot) reject it, because at prefill time there is no tool
call yet.

## Root cause (llama.cpp side)

With `return_prefill: true`, llama-server correctly echoes the prefilled
`reasoning_content` and continues generation, but the **tool-call parser is not
applied to the post-prefill output**. The model produces the tool-call tokens
(`<function=NAME>...</function>` for Qwen3), but they are returned
verbatim in `content` with `finish_reason: "stop"` instead of being parsed into
`message.tool_calls` with `finish_reason: "tool_calls"`. The `__verbose` payload
even shows the tool-call GBNF grammar is attached, yet the result is not parsed
into `tool_calls`. This is a `return_prefill` + tools interaction bug in
llama-server's chat-completions handler, not a pi bug.

Independent secondary issue: prefilling an assistant message whose `tool_calls`
field is already populated (cases 1 and 2) is rejected by the Qwen3 chat
template's `multi_step_tool` branch with `Jinja Exception: No user query found
in messages.` before generation even starts. That path is currently moot for pi
(it rejects tool-call prefills client-side), but it is another gap if
continuation of an interrupted tool call were ever desired.

## Test cases

All against `http://localhost:5021`. The model emits Qwen3-style tool calls:
`<tool_call>\n<function=NAME>\n<parameter=KEY>VALUE\n</parameter>\n</function>\n</tool_call>`
(rendered by the chat template). A `/tmp/sample.txt` (`hello world`) and
`/tmp/sample2.txt` (`second file`) are used as read fixtures. `tool_choice:
"auto"` plus a `tools` array is the only "magic" needed -- llama-server renders
the tools into the chat template and parses tool-call tokens server-side.

### Case 4 -- baseline: successful tool call, NO prefill (control)

Works correctly: returns structured `tool_calls`, `finish_reason: "tool_calls"`.

```bash
cat > /tmp/c4-baseline.json <<'EOF'
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools. When you need information, call the appropriate tool. After receiving tool results, answer the user."},
    {"role": "user", "content": "Read the file at /tmp/sample.txt and tell me its contents."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "read_file", "description": "Read the contents of a file at the given path.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute path to the file to read."}}, "required": ["path"]}}}
  ],
  "tool_choice": "auto",
  "max_tokens": 512,
  "stream": false
}
EOF
curl -s http://localhost:5021/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/c4-baseline.json | python3 -m json.tool
```

Observed: `finish_reason: "tool_calls"`, `message.tool_calls[0].function.name ==
"read_file"`, `arguments == {"path":"/tmp/sample.txt"}`, `content: ""`,
`reasoning_content` populated. `__verbose.content` contains the raw
`<function=read_file>...` markers (proving the grammar parses them
server-side into `tool_calls` when `return_prefill` is absent).

### Case 3 -- reasoning prefill only (the reported bug)

Prefill an assistant message with `reasoning_content` only (no `tool_calls`),
priming the model to call the tool. **This is the failing real-world path.**

```bash
cat > /tmp/c3-reasoning-prefill.json <<'EOF'
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools. When you need information, call the appropriate tool. After receiving tool results, answer the user."},
    {"role": "user", "content": "Read the file at /tmp/sample.txt and tell me its contents."},
    {"role": "assistant", "reasoning_content": "The user wants to read the contents of a file at `/tmp/sample.txt`.\nI should use the `read_file` tool with the path `/tmp/sample.txt`."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "read_file", "description": "Read the contents of a file at the given path.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute path to the file to read."}}, "required": ["path"]}}}
  ],
  "tool_choice": "auto",
  "max_tokens": 512,
  "stream": false,
  "return_prefill": true
}
EOF
curl -s http://localhost:5021/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/c3-reasoning-prefill.json | python3 -m json.tool
```

Observed (non-stream and stream both reproduce):

- `finish_reason: "stop"` (should be `"tool_calls"`)
- `message.content` = the raw tool-call markers as plain text
  (`<tool_call>\n<function=read_file>\n<parameter=path>/tmp/sample.txt\n</parameter>\n</function>\n</tool_call>`)
- `message.tool_calls` = `null` (absent)
- `message.reasoning_content` = the prefill echoed, followed by a few more
  reasoning lines (echo + continuation works fine)

Streaming variant (add `"stream": true`, `"stream_options": {"include_usage":
true}`): deltas arrive as `content` text only, no `tool_calls` deltas,
`finish_reason: "stop"`. This is the exact path pi hits.

### Case 1 -- tool_calls prefill, single (expected to fail)

Prefill an assistant message that already carries a `tool_calls` entry, with
`return_prefill: true`. Tests whether llama-server can echo a prefilled tool
call and continue. pi rejects this client-side (`_canPrefillContinue` returns
false for tool-call messages), so this is a llama-server capability probe.

```bash
cat > /tmp/c1-toolcalls-prefill.json <<'EOF'
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools. When you need information, call the appropriate tool. After receiving tool results, answer the user."},
    {"role": "user", "content": "Read the file at /tmp/sample.txt and tell me its contents."},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"/tmp/sample.txt\"}"}}
    ]}
  ],
  "tools": [
    {"type": "function", "function": {"name": "read_file", "description": "Read the contents of a file at the given path.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute path to the file to read."}}, "required": ["path"]}}}
  ],
  "tool_choice": "auto",
  "max_tokens": 256,
  "stream": false,
  "return_prefill": true
}
EOF
curl -s http://localhost:5021/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/c1-toolcalls-prefill.json | python3 -m json.tool
```

Observed: HTTP 400, `{"error":{"code":400,"message":"Unable to generate parser
for this template. Automatic parser generation failed: ... Jinja Exception: No
user query found in messages."}}`. The Qwen3 chat template's `multi_step_tool`
branch raises before any generation. The trailing assistant `tool_calls` message
with no following tool result trips the template.

### Case 2 -- tool_calls prefill, multiple (expected to fail)

Same as case 1 but with two tool calls in the prefill.

```bash
cat > /tmp/c2-multi-toolcalls-prefill.json <<'EOF'
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools. When you need information, call the appropriate tool. After receiving tool results, answer the user."},
    {"role": "user", "content": "Read the files at /tmp/sample.txt and /tmp/sample2.txt, then compare them."},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"/tmp/sample.txt\"}"}},
      {"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"/tmp/sample2.txt\"}"}}
    ]}
  ],
  "tools": [
    {"type": "function", "function": {"name": "read_file", "description": "Read the contents of a file at the given path.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Absolute path to the file to read."}}, "required": ["path"]}}}
  ],
  "tool_choice": "auto",
  "max_tokens": 256,
  "stream": false,
  "return_prefill": true
}
EOF
curl -s http://localhost:5021/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/c2-multi-toolcalls-prefill.json | python3 -m json.tool
```

Observed: same HTTP 400 `Jinja Exception: No user query found in messages.` as
case 1.

## Summary table

| Case | Prefill | `return_prefill` | Result | finish_reason |
|------|---------|------------------|--------|---------------|
| 4 | none | no | structured `tool_calls` | `tool_calls` (correct) |
| 3 | `reasoning_content` only | yes | raw tool text in `content`, no `tool_calls` | `stop` (BUG) |
| 1 | `tool_calls` (single) | yes | HTTP 400 template error | n/a |
| 2 | `tool_calls` (multiple) | yes | HTTP 400 template error | n/a |

## What needs fixing in llama.cpp

Case 3 is the bug that breaks pi's `/continue` after an interrupted thought:
when `return_prefill: true` is set and tools are present, llama-server must run
the post-prefill generation output through the same tool-call parser it uses for
the non-prefill path, so tool-call tokens are returned as structured
`message.tool_calls` with `finish_reason: "tool_calls"` instead of leaking into
`content` with `finish_reason: "stop"`.

Cases 1 and 2 (prefilling an assistant message that already has `tool_calls`)
are a separate, lower-priority issue: the chat template rejects the message
shape before generation. pi never sends this shape (it rejects tool-call
prefills client-side), so fixing it is only needed if continuation of an
interrupted-in-progress tool call were ever desired.

## Reproduce the fixtures

```bash
echo "hello world" > /tmp/sample.txt
echo "second file" > /tmp/sample2.txt
```
