# Plan: Tool call prefill support (continuation from previous session)

## Context

The previous session fixed tool call **parsing** with reasoning/content prefill
(commit `f6709a03c`). Cases 1–4 (reasoning/content prefill + tool call
generation) now work: the PEG parser processes `generation_prompt_for_parser +
generated_text` and correctly extracts structured `tool_calls`.

**What remains:** supporting prefill of **partial tool calls** so a client
can resume an interrupted tool call. The client receives structured
`tool_calls` deltas (with `function.name` and `function.arguments`), but the
special tokens are stripped. The client cannot reconstruct the raw model-format
tokens to send back as a prefill.

## Design decision

**Do NOT try to reverse structured `tool_calls` back to raw tokens.** That
path is fragile (JSON-to-tag conversion, partial JSON handling, per-model
format differences).

Instead:

1. **Add a raw token field to `tool_calls` streaming deltas.** Each
   `tool_calls` delta in the stream includes a `__raw` string containing the
   exact raw model-format text that was parsed for that delta. The client
   accumulates `__raw` across deltas to get the full raw tool-call text.

2. **Add a `tool_calls_raw` prefill field** (string, not array). When the
   trailing assistant message has `tool_calls_raw` (not null, even if empty),
   it is treated as raw model-format text that is appended to
   `generation_prompt_for_parser` after reasoning/content. The PEG parser
   parses `generation_prompt_for_parser + generated_text` and completes the
   tool call. `content` is considered closed when `tool_calls_raw` is present.

3. **Structured `tool_calls` in the trailing assistant message** (the
   existing OpenAI field) is supported only for **complete** tool calls. The
   server serializes them to raw text via `common_chat_tool_calls_to_text`
   (already implemented in `common/chat.cpp`). Partial structured tool calls
   are not supported via the `tool_calls` field — the client must use
   `tool_calls_raw` for partial tool calls.

## What was already implemented in the previous session

### Committed (working)

- `common/chat.h`: Added `common_chat_tool_format` fields to
  `common_chat_params` (tool format mode, section/per-call markers, name/arg
  markers, etc.).
- `common/chat.cpp`: Added `common_chat_tool_calls_to_text()` function that
  serializes structured `tool_calls` to raw model-format text using the tool
  format info. Handles `JSON_NATIVE`, `TAG_WITH_JSON`, `TAG_WITH_TAGGED`
  formats. Includes partial JSON fallback for `TAG_WITH_TAGGED`.
- `common/chat.cpp`: Populated tool format fields from `autoparser` in
  `common_chat_templates_apply_jinja`.
- `tools/server/server-common.cpp`: Construct `generation_prompt_for_parser`
  for prefill cases 1–4 using `thinking_start_tag`/`thinking_end_tag`.
  Set `__prefill_is_continuation = true`.
- `tools/server/server-schema.cpp`: Handle `__prefill_is_continuation` to set
  `chat_parser_params.is_continuation`.
- `tools/server/server-task.cpp`: Use `common_chat_parse` for all prefill
  cases (removed manual splitting). Adjusted `return_prefill` handling
  (subtract prefill when false, emit as first delta when true for streaming).
- `tools/server/server-task.h`: Removed `prefill_opening_sequence`/
  `prefill_closing_sequence` (no longer needed).
- `tools/server/server-context.cpp`: Removed prefill opening/closing sequence
  computation.

### NOT yet implemented

- `__raw` field in streaming `tool_calls` deltas.
- `tool_calls_raw` prefill field detection and handling.
- Complete `tool_calls` prefill via serialization (detect `tool_calls` in
  trailing assistant message, serialize to raw text, append to
  `generation_prompt_for_parser`).
- `return_prefill` handling for tool calls (exact prefill must be sent back;
  see "return_prefill semantics" below).

## Implementation steps

### Step 1: Add `__raw` to streaming `tool_calls` deltas

**File: `tools/server/server-task.cpp`**

In `server_chat_msg_diff_to_json_oaicompat()` (or wherever tool call deltas
are serialized to JSON), add a `__raw` field to each tool call delta. The
raw text is the text that was parsed for that delta.

The raw text for a tool call delta is not directly available in the
`common_chat_msg_diff` struct. We need to track it. Options:

- **Option A:** Track raw text in `task_result_state::generated_text`. The
  `generated_text` is `generation_prompt_for_parser + accumulated_text`. The
  tool call raw text is a substring of `generated_text`. We can track the
  position of the tool call in `generated_text` and extract the raw text.

- **Option B (simpler):** In `update_chat_msg`, after parsing, compute the
  raw tool-call text by re-serializing the parsed `tool_calls` using
  `common_chat_tool_calls_to_text`. But this gives the complete raw text, not
  per-delta. For per-delta, we can diff the raw text between the previous and
  current `chat_msg`.

- **Option C (simplest, recommended):** In the final (non-streaming)
  response, include the full raw tool-call text in `__verbose` (already
  present as `content`). For streaming, include the accumulated raw text in a
  `__raw` field on each `tool_calls` delta. The accumulated raw text is the
  full raw text up to that point (not per-delta, but cumulative). The client
  can diff it to get per-delta raw text, or just use the final value.

  Actually, the simplest approach: add `__raw` to the **non-streaming**
  response's `tool_calls` array (each tool call gets a `__raw` field with the
  full raw text). For streaming, add `__raw` to the final chunk's
  `tool_calls` (the last delta before `finish_reason`). The client
  accumulates structured deltas during streaming and gets the full raw text
  from the final chunk.

  **Even simpler:** Since the `__verbose.content` field already contains the
  full raw model output (including tool call tokens), the client can extract
  the tool call raw text from there. But the user said `__verbose` is for
  debugging only. So we need a proper `__raw` field.

  **Recommended approach:** Add `__raw` to each tool call in the
  **non-streaming** response's `tool_calls` array. Each tool call object gets
  a `__raw` field with the full raw text for that tool call. For streaming,
  the final `finish_reason` chunk includes the full `tool_calls` array with
  `__raw` fields. The client uses `__raw` for prefill.

  To get the raw text per tool call, we need to track which part of
  `generated_text` corresponds to each tool call. The PEG parser doesn't
  provide this directly. We can compute it by re-serializing each parsed
  tool call using `common_chat_tool_calls_to_text` and using that as `__raw`.
  This gives the complete raw text for a complete tool call. For a partial
  tool call (mid-stream), the re-serialized text would be the partial raw
  text (since the tool call is incomplete).

  Wait — re-serializing a parsed tool call gives the **complete** raw text
  (the parser normalizes it). For a partial tool call mid-stream, the parsed
  tool call has partial arguments, and re-serializing gives the partial raw
  text. This is exactly what we want for prefill!

  So: `__raw = common_chat_tool_calls_to_text({tc}, chat_params)` for each
  tool call `tc` in the parsed message. This gives the exact raw text for
  that tool call, which the client can use for prefill.

  **Implementation:**
  - In `common_chat_msg::to_json_oaicompat()`, add `__raw` field to each
    tool call in the output. But `to_json_oaicompat` doesn't have access to
    `chat_params`. We need to pass `chat_params` or the tool format info.
  - Alternative: add `__raw` in the server (in `to_json_oaicompat_chat` or
    `update`) where `chat_params` / tool format info is available.
  - The tool format info is in `chat_parser_params` (we need to add it there)
    or in the prefill state.

  **Actually, the tool format info needs to be available in
  `task_result_state` (for `update_chat_msg`) and in the result structs (for
  `to_json_oaicompat_chat`).** We need to pass the tool format info from
  `common_chat_params` through `llama_params` to the server.

### Step 2: Pass tool format info through `llama_params`

**File: `tools/server/server-common.cpp`**

After `common_chat_templates_apply`, pass the tool format fields to
`llama_params`:

```cpp
llama_params["__tool_format_mode"] = chat_params.tool_format_mode;
llama_params["__tool_section_start"] = chat_params.tool_section_start;
llama_params["__tool_section_end"] = chat_params.tool_section_end;
// ... all tool format fields ...
```

**File: `tools/server/server-schema.cpp`**

Add field handlers to read the `__tool_*` fields from `llama_params` and
store them in `chat_parser_params` (or a new struct in `task_params`).

**File: `common/chat.h`**

Add tool format fields to `common_chat_parser_params` (or create a new
`common_chat_tool_format` struct and include it in
`common_chat_parser_params`).

### Step 3: Add `__raw` to tool call output

**File: `tools/server/server-task.cpp`**

In `to_json_oaicompat_chat()` (non-streaming) and
`to_json_oaicompat_chat_stream()` (streaming), add `__raw` to each tool call
using `common_chat_tool_calls_to_text({tc}, tool_format_params)`.

For streaming, add `__raw` to each tool call delta. The `__raw` for a delta
is the cumulative raw text for that tool call up to that point. The client
can use the final `__raw` value for prefill.

Actually, for streaming, the simplest approach is: in the final
`finish_reason` chunk, include the full `tool_calls` array with `__raw`
fields. During streaming, the `tool_calls` deltas have `function.name` and
`function.arguments` (structured), and the final chunk has `__raw`.

But the user wants `__raw` on each streaming line (delta). Let me add it to
each tool call delta. The `__raw` for a delta is the raw text for the
tool_call delta. Since the PEG parser doesn't provide per-delta raw text, I
can compute it by re-serializing the parsed tool call up to that point.

Actually, the simplest correct approach: in `update_chat_msg`, after
parsing, re-serialize each tool call using `common_chat_tool_calls_to_text`
and store the raw text in the `common_chat_msg_diff`. Add a `__raw` field to
`common_chat_msg_diff` for tool call deltas.

**File: `common/chat.h`**

Add `std::string raw` field to `common_chat_tool_call` (or to
`common_chat_msg_diff`'s `tool_call_delta`).

**File: `common/chat.cpp`**

In `common_chat_msg_diff::compute_diffs`, compute the raw text for each tool
call diff using `common_chat_tool_calls_to_text`.

Actually, this is getting complex. Let me simplify: **add `__raw` to the
non-streaming response only.** The client gets the full raw text for each
tool call from the non-streaming response. For streaming, the client can use
the structured deltas and the final `__raw` from the non-streaming response
(which is included in the last chunk). Or the client can wait for the
non-streaming response.

**Even simpler:** add `__raw` to `common_chat_tool_call` (the parsed tool
call struct). The PEG parser fills it during parsing. Then it's available in
both streaming and non-streaming responses.

But the PEG parser doesn't currently fill a `__raw` field. We'd need to
modify the PEG parser to capture the raw text for each tool call. This is
doable but requires changes to the PEG parser.

**Simplest viable approach for now:** In the server, after parsing, compute
`__raw` for each tool call by calling
`common_chat_tool_calls_to_text({tc}, tool_format_params)`. Add it to the
JSON output. This gives the complete raw text for each tool call. For
partial tool calls (mid-stream), the re-serialized text is the partial raw
text (since the tool call is incomplete). This is exactly what the client
needs for prefill.

### Step 4: Detect `tool_calls_raw` prefill field

**File: `tools/server/server-common.cpp`**

In `oaicompat_chat_params_parse`, after the existing prefill detection:

```cpp
// Check for tool_calls_raw in the trailing assistant message
std::string prefill_tool_calls_raw;
bool has_tool_calls_raw = false;
if (prefill_assistant_message) {
    const auto & last_json_msg = messages.back();
    if (last_json_msg.contains("tool_calls_raw") && !last_json_msg.at("tool_calls_raw").is_null()) {
        has_tool_calls_raw = true;
        prefill_tool_calls_raw = last_json_msg.at("tool_calls_raw").get<std::string>();
    }
}
```

If `has_tool_calls_raw` is true:
- `content` is considered closed (even if empty)
- Append `prefill_tool_calls_raw` to `generation_prompt_for_parser`
- Set `prefill_case` to a new value (e.g., 5 for tool_calls_raw)
- Pass `prefill_tool_calls_raw` to the slot for `return_prefill` handling

The `generation_prompt_for_parser` construction:
```cpp
if (has_tool_calls_raw) {
    // Append tool_calls_raw to the generation_prompt_for_parser
    // The PEG parser will parse the raw tool call text + generated text
    generation_prompt_for_parser += prefill_tool_calls_raw;
}
```

This works for all cases:
- Case 1 (reasoning only) + tool_calls_raw:
  `generation_prompt_for_parser = gen_prompt + thinking_start_suffix + reasoning + tool_calls_raw`
  Wait, for case 1, the reasoning block is open (no thinking_end). But tool_calls_raw
  comes after content, which comes after reasoning. So we need to close the reasoning
  block first. Actually, if tool_calls_raw is present, the reasoning should be closed
  (the model is now generating tool calls, not reasoning). So:
  `generation_prompt_for_parser = gen_prompt + thinking_start_suffix + reasoning + thinking_end + tool_calls_raw`

  But for case 1 (reasoning only, content null), the reasoning block is open. If
  tool_calls_raw is present, we should close the reasoning block and then append
  tool_calls_raw. So treat it like case 4 (reasoning + empty content) + tool_calls_raw.

- Case 2 (reasoning + content) + tool_calls_raw:
  `generation_prompt_for_parser = gen_prompt + thinking_start_suffix + reasoning + thinking_end + content + tool_calls_raw`

- Case 3 (content only) + tool_calls_raw:
  `generation_prompt_for_parser = assistant_prefix + content + tool_calls_raw`

- Case 4 (reasoning + empty content) + tool_calls_raw:
  `generation_prompt_for_parser = gen_prompt + thinking_start_suffix + reasoning + thinking_end + tool_calls_raw`

- No reasoning, no content, only tool_calls_raw:
  `generation_prompt_for_parser = assistant_prefix + tool_calls_raw`

### Step 5: Detect complete `tool_calls` prefill

**File: `tools/server/server-common.cpp`**

If the trailing assistant message has `tool_calls` (structured, not
`tool_calls_raw`), serialize them to raw text using
`common_chat_tool_calls_to_text` and treat as `tool_calls_raw`:

```cpp
if (last_msg.tool_calls.empty() == false) {
    std::string raw = common_chat_tool_calls_to_text(last_msg.tool_calls, chat_params);
    if (!raw.empty()) {
        has_tool_calls_raw = true;
        prefill_tool_calls_raw = raw;
    }
}
```

This handles complete tool calls. Partial tool calls must use `tool_calls_raw`.

### Step 6: `return_prefill` semantics for tool calls

**Key rule:** For `return_prefill = true`, the response must include the
EXACT prefilled tool calls (as the client sent them), not a re-parsed
version. For `return_prefill = false`, the response must include only the
model's generated contribution.

With `is_continuation = true`, the PEG parser processes
`generation_prompt_for_parser + generated_text`. The parsed `oaicompat_msg`
includes both prefilled and generated tool calls. The parser may normalize
the prefilled tool calls (e.g., JSON formatting), so they might not match
the exact prefill.

**For `return_prefill = true`:**
- The streaming diffs (from `is_continuation = true`) exclude the prefill
  (the initial `chat_msg` has the prefill, and diffs are computed from it).
- For streaming, emit the exact prefilled tool calls as the first delta(s).
- For non-streaming, replace the first N tool calls in `oaicompat_msg` with
  the exact prefilled tool calls from the request. Keep the remaining tool
  calls (generated) as-is.

  But the model might have COMPLETED a prefilled tool call (added more
  arguments). The completed tool call has the full arguments (prefill +
  generated). Replacing it with the exact prefilled tool call would lose the
  model's contribution. So for completed tool calls, we need to merge: use
  the exact prefilled tool call's name, but update the arguments with the
  model's completion.

  Actually, this is complex. Simpler approach: for `return_prefill = true`,
  send the exact prefilled tool calls as separate deltas (before the generated
  deltas), and the generated deltas include the model's completions/additions.
  The client reconstructs the full tool calls by applying the generated deltas
  on top of the prefilled tool calls.

  This is exactly what `is_continuation = true` does for streaming: the
  initial `chat_msg` has the prefilled tool calls, and the diffs are the
  model's additions. The diffs include:
  - For completed tool calls: the additional arguments (delta on the
    prefilled tool call)
  - For new tool calls: the complete new tool calls

  So for streaming with `return_prefill = true`:
  - Emit the exact prefilled tool calls as the first delta(s)
  - Then emit the generated deltas (which are the model's additions)

  For streaming with `return_prefill = false`:
  - Emit only the generated deltas (the model's additions)

  For non-streaming with `return_prefill = true`:
  - The `oaicompat_msg` has all tool calls (prefilled + generated). Replace
    the first N tool calls with the exact prefilled tool calls, but keep the
    arguments from the parser (which include the model's completions).

    Actually, the `oaicompat_msg` tool calls have the full arguments
    (prefill + generated) because the parser processes
    `generation_prompt_for_parser + generated_text` and the
    `generation_prompt_for_parser` includes the prefilled tool calls. So the
    parsed tool calls have the complete arguments. For `return_prefill = true`,
    the response should include these complete tool calls. But the tool call
    `name` and `id` might be normalized by the parser. The user wants the
    EXACT prefill, so we should use the exact prefilled `name` and `id`, but
    the parser's `arguments` (which include the model's completions).

    Hmm, this is getting complex. Let me simplify:

    For `return_prefill = true`, non-streaming:
    - The `oaicompat_msg` has all tool calls with complete arguments
    - The response includes all tool calls as-is (the parser's output)
    - The exact prefill is already in the response (the parser processed it)
    - BUT the parser might normalize the tool call format (e.g., JSON
      whitespace). The user wants the EXACT prefill.

    The user said "You have to send exact prefill." This means the tool call
    `name`, `id`, and `arguments` in the response must match the prefill
    exactly. The parser might normalize the arguments (e.g., add spaces to
    JSON). So we need to replace the parser's tool calls with the exact
    prefilled tool calls for the prefilled part.

    OK, here's the approach:
    - Store the exact prefilled tool calls (from the request)
    - For `return_prefill = true`, non-streaming: the response includes the
      exact prefilled tool calls + the parser's generated tool calls (the ones
      beyond the prefilled count). For completed tool calls (where the model
      added to a prefilled tool call), the response includes the exact
      prefilled tool call with updated arguments (the parser's arguments).
    - For `return_prefill = false`, non-streaming: the response includes only
      the parser's generated tool calls (the ones beyond the prefilled count).
      For completed tool calls, the response includes the model's addition
      (the diff between the completed and prefilled arguments).

    Actually, the simplest correct approach for non-streaming:
    - Store the exact prefilled tool calls (from the request) and the count N
    - For `return_prefill = true`: use the exact prefilled tool calls for the
      first N, and the parser's tool calls for the rest. For completed tool
      calls (same index, same name, more arguments), use the exact prefilled
      name/id but the parser's arguments (which include the model's
      completions).
    - For `return_prefill = false`: use only the parser's tool calls beyond
      index N. For completed tool calls (index < N), include the model's
      addition (the diff between the parser's arguments and the prefilled
      arguments). But computing this diff is complex.

    **Even simpler (and correct) approach:**
    - For `return_prefill = true`: emit the exact prefilled tool calls as
      initial deltas (streaming) or include them in the response
      (non-streaming). Then emit/include the parser's generated deltas/tool
      calls. The client applies the generated deltas on top of the prefilled
      tool calls.
    - For `return_prefill = false`: emit/include only the parser's generated
      deltas/tool calls. The client has the prefilled tool calls from the
      request and applies the generated deltas.

    For streaming, this is natural: the `is_continuation = true` flag
    initializes `chat_msg` with the parsed prefill, and the diffs are the
    model's additions. For `return_prefill = true`, emit the exact prefilled
    tool calls as the first delta(s). For `return_prefill = false`, the diffs
    are the model's additions (no prefill).

    For non-streaming, the `oaicompat_msg` has all tool calls. For
    `return_prefill = true`, include the exact prefilled tool calls + the
    parser's generated tool calls. For `return_prefill = false`, include only
    the parser's generated tool calls.

    But how to distinguish prefilled from generated in `oaicompat_msg`? The
    first N tool calls are prefilled (possibly completed by the model), and
    the rest are generated. For `return_prefill = true`, replace the first N
    with the exact prefilled tool calls (keeping the parser's arguments if the
    model completed them). For `return_prefill = false`, remove the first N.

    Wait, the user said removing the first N is wrong because the model might
    have completed them. For `return_prefill = false`, the client has the
    prefilled tool calls and wants only the model's additions. The model's
    additions for a completed tool call are the additional arguments. But the
    `oaicompat_msg` has the complete arguments (prefill + generated). Removing
    the first N tool calls would lose the model's additions.

    So for `return_prefill = false`, I need to:
    1. For the first N tool calls (prefilled): compute the diff between the
       parser's arguments and the prefilled arguments. The diff is the model's
       addition. Include the tool call with the diff arguments (not the full
       arguments).
    2. For tool calls beyond N: include them as-is (they are generated).

    The diff between the parser's arguments and the prefilled arguments is a
    string suffix (the parser's arguments start with the prefilled arguments
    plus the model's additions). So the model's addition is
    `parser_arguments.substr(prefilled_arguments.size())`.

    But this only works if the parser's arguments start with the prefilled
    arguments exactly. The parser might normalize the JSON (e.g., add spaces),
    so the prefix might not match. This is the same issue as with
    reasoning/content prefix subtraction.

    For reasoning/content, I use `find(prefill) == 0` to check if the
    parser's output starts with the prefill. If it does, I subtract the
    prefill. If not, I don't subtract (the parser normalized it).

    For tool call arguments, I can do the same: check if the parser's
    arguments start with the prefilled arguments, and if so, subtract the
    prefill. If not, include the full arguments (the parser normalized it, so
    the prefill is lost).

    **Final approach for `return_prefill` with tool calls:**
    - Store the exact prefilled tool calls and count N
    - For `return_prefill = true`:
      - Streaming: emit exact prefilled tool calls as first delta(s), then
        generated deltas
      - Non-streaming: for the first N tool calls, use the exact prefilled
        name/id but the parser's arguments (which include model's completions).
        For tool calls beyond N, use the parser's tool calls as-is.
    - For `return_prefill = false`:
      - Streaming: emit only generated deltas (is_continuation handles this)
      - Non-streaming: for the first N tool calls, subtract the prefilled
        arguments from the parser's arguments (if prefix matches). For tool
        calls beyond N, use the parser's tool calls as-is.

### Step 7: Pass tool format info through the server

**File: `tools/server/server-common.cpp`**

Pass tool format fields from `chat_params` to `llama_params`:
```cpp
llama_params["__tool_format_mode"] = chat_params.tool_format_mode;
llama_params["__tool_section_start"] = chat_params.tool_section_start;
// ... all tool format fields ...
```

**File: `tools/server/server-schema.cpp`**

Add field handlers for the `__tool_*` fields and store them in
`task_params` (or a new struct).

**File: `tools/server/server-task.h`**

Add tool format fields to `task_result_state` and result structs (for
`common_chat_tool_calls_to_text` in the server).

**File: `tools/server/server-context.cpp`**

Pass tool format fields from `task.data` to the slot and result structs.

### Step 8: Use `common_chat_tool_calls_to_text` for `__raw` and complete `tool_calls` prefill

**File: `tools/server/server-task.cpp`**

In `to_json_oaicompat_chat()` and `to_json_oaicompat_chat_stream()`, add
`__raw` to each tool call using `common_chat_tool_calls_to_text`.

**File: `tools/server/server-common.cpp`**

In the prefill detection, handle `tool_calls` (complete) and `tool_calls_raw`
(partial/complete raw text):
- `tool_calls` (complete): serialize via `common_chat_tool_calls_to_text`
- `tool_calls_raw` (raw text): use as-is
- Append to `generation_prompt_for_parser`

### Step 9: Handle `tool_calls_raw` in `common_chat_msgs_parse_oaicompat`

**File: `common/chat.cpp`**

In `common_chat_msgs_parse_oaicompat`, accept `tool_calls_raw` in the
assistant message (as a string field, not part of the standard OpenAI spec).

### Step 10: `return_prefill` handling for tool calls

**File: `tools/server/server-task.h`**

Add `prefill_tool_calls_raw` and `prefill_tool_calls_count` to
`task_result_state` and result structs.

**File: `tools/server/server-task.cpp`**

In `server_task_result_cmpl_final::update()`:
- For `return_prefill = false`: subtract prefilled tool call arguments
- For `return_prefill = true`: replace first N tool calls with exact prefill

In `server_task_result_cmpl_partial::update()`:
- For `return_prefill = true`: emit exact prefilled tool calls as first delta(s)

## Summary of files to edit

1. `common/chat.h` — add tool format fields to `common_chat_params` (done),
   add `__raw` field to `common_chat_tool_call` or output, add
   `tool_calls_raw` support to `common_chat_msg`
2. `common/chat.cpp` — `common_chat_tool_calls_to_text` (done), populate
   tool format fields (done), accept `tool_calls_raw` in
   `common_chat_msgs_parse_oaicompat`
3. `tools/server/server-common.cpp` — pass tool format fields to `llama_params`,
   detect `tool_calls`/`tool_calls_raw` prefill, serialize complete `tool_calls`,
   append to `generation_prompt_for_parser`
4. `tools/server/server-schema.cpp` — handle `__tool_*` fields, `__prefill_*`
   fields
5. `tools/server/server-task.h` — add tool format and prefill tool call fields
   to structs
6. `tools/server/server-task.cpp` — `__raw` in output, `return_prefill`
   handling for tool calls
7. `tools/server/server-context.cpp` — pass tool format and prefill tool call
   fields to slot/result

## Testing

After implementation, test with:

1. **Complete `tool_calls` prefill**: trailing assistant message with
   `tool_calls` (complete) + `return_prefill: true`. Model should not
   regenerate the tool call.

2. **`tool_calls_raw` prefill (partial)**: trailing assistant message with
   `tool_calls_raw` (partial raw text from `__raw` field). Model should
   complete the tool call.

3. **`return_prefill: false` with tool calls**: response should include only
   the model's additions (not the prefilled tool calls).

4. **`return_prefill: true` with tool calls**: response should include the
   exact prefilled tool calls + the model's additions.

5. **Multiple tool calls**: one complete + one partial. Model should complete
   the partial one and both should be in the response.
