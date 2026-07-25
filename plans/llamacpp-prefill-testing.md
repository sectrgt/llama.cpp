# Testing prefill / continuation with tool calls

All tests use `curl` against a running `llama-server` (here at `localhost:5021`).
Adjust the `--model`, `--port`, and tool definitions as needed for your model.

## Fixtures

```bash
echo "hello world" > /tmp/sample.txt
echo "second file" > /tmp/sample2.txt
```

## Prefill cases

The server detects four prefill cases based on the trailing assistant message:

| Case | `reasoning_content` | `content` | Behavior |
|------|---------------------|-----------|----------|
| 1    | non-empty           | null/absent | Continue reasoning (open thinking block) |
| 2    | non-empty           | non-empty | Close reasoning, continue content |
| 3    | null/absent         | non-empty | Content-only prefill (no reasoning) |
| 4    | non-empty           | `""` (empty string) | Close reasoning, generate content from scratch |

**Key rule:** `content: ""` (empty string) closes the reasoning block and starts
content from scratch. `content: null` (or absent) keeps the reasoning block open.

## `return_prefill` option

- `return_prefill: false` (default): response contains **only** the newly
  generated text (prefill is stripped).
- `return_prefill: true`: response contains **prefill + generated** text.

## Case 4 — baseline: tool call, no prefill (control)

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt and tell me its contents."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"`, `tool_calls[0].function.name == "read_file"`.

## Case 3 — reasoning prefill + tool call (the reported bug)

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt and tell me its contents."},
      {"role": "assistant", "reasoning_content": "I should use the read_file tool."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"`, `tool_calls` populated, `reasoning_content`
includes the prefill. With `return_prefill: false`, `reasoning_content` contains
only the model's generated reasoning (prefill stripped).

## Case 1 — reasoning only (open thinking block) + tool call

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "reasoning_content": "I need to read the file."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"`, `tool_calls` populated. The model
continues reasoning (open block) then emits a tool call.

## Case 2 — reasoning + content prefill + tool call

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "reasoning_content": "I need to read the file.", "content": "I will now read the file."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"`, `tool_calls` populated, `reasoning_content`
includes prefill, `content` includes prefill + generated. With `return_prefill:
false`, both are stripped to only the generated portion.

## Case 4 — reasoning + empty content (closed reasoning) + tool call

```bash
# Without trailing newline in reasoning_content:
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "reasoning_content": "I need to read the file.", "content": ""}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false
  }' | python3 -m json.tool

# With trailing newline in reasoning_content:
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "reasoning_content": "I need to read the file.\n", "content": ""}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"`, `tool_calls` populated. With
`return_prefill: false`, `reasoning_content` should be empty (no stray newline)
regardless of whether the prefilled reasoning had a trailing newline.

## Case 3 — content-only prefill (no reasoning)

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "The answer is"}
    ],
    "max_tokens": 32, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `content: "The answer is 4."` (prefill + generated). With
`return_prefill: false`, `content: " 4."` (generated only).

## Streaming tests

Add `"stream": true, "stream_options": {"include_usage": true}` to any request
and pipe through a parser that shows each delta. Verify:

- With `return_prefill: true`: the first delta(s) contain the prefilled
  reasoning/content, followed by the generated deltas.
- With `return_prefill: false` (default): the first delta(s) contain only the
  generated text (no prefill).
- Tool call deltas arrive as structured `tool_calls` with `function.name` and
  incremental `function.arguments`.

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "reasoning_content": "I should use the read_file tool."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": true, "return_prefill": true
  }' > /tmp/stream-output.txt 2>&1

# Show tool_calls deltas and finish_reason
python3 -c "
import json
for line in open('/tmp/stream-output.txt'):
    if not line.startswith('data: '): continue
    data = line[6:].strip()
    if data == '[DONE]': continue
    obj = json.loads(data)
    for c in obj.get('choices', []):
        d = c.get('delta', {})
        if 'tool_calls' in d:
            print('tool_calls:', json.dumps(d['tool_calls']))
        if 'reasoning_content' in d and d['reasoning_content']:
            print('reasoning:', repr(d['reasoning_content'][:60]))
        if 'content' in d and d['content']:
            print('content:', repr(d['content'][:60]))
        if c.get('finish_reason'):
            print('finish_reason:', c['finish_reason'])
"
```

## Partial tool call prefill (future work)

Tool calls are streamed as structured `tool_calls` deltas (special tokens are
stripped by the PEG parser). To prefill a partial tool call, the server needs
the raw model-format text (with special tokens), which the client does not have.

Two approaches are under consideration:

1. **Server-side serialization**: Make the auto-parser's tool format info
   (`tool_format_analysis`) available in `common_chat_params`, pass it to the
   server, and serialize structured `tool_calls` back to raw model-format text
   for the `generation_prompt_for_parser`. This requires a serializer for each
   format type (`JSON_NATIVE`, `TAG_WITH_JSON`, `TAG_WITH_TAGGED`) and handling
   partial `function.arguments` JSON strings.

2. **Raw text in streaming deltas**: Add a field (e.g. `__raw`) to each
   `tool_calls` streaming delta containing the exact raw tokens. The client
   accumulates this and sends it back in the `content` field for prefill. This
   is model-agnostic but requires an API change.

Proposed prefill semantics for `tool_calls`:
- Three prefill fields: `reasoning_content`, `content`, `tool_calls`
- If `tool_calls` is present (not null, even if empty), `content` is considered
  closed and the model should focus on completing the tool call
- The server serializes `tool_calls` to raw text and includes it in
  `generation_prompt_for_parser` so the PEG parser can complete the tool call
