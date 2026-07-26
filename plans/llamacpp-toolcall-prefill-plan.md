# Plan: tool-call raw-token prefill support

## Goal

Let a client resume an interrupted assistant message that ends inside a tool
call. The client must receive the **exact raw model-format tokens** that were
parsed into each structured `tool_calls` entry, so it can send them back as a
prefill and the model continues from the exact same token sequence.

## Design principles

1. **Models consume tokens, not structured data.** The response must expose
the raw token string for each tool call (`__raw`). This is the same text that
appears in `__verbose.content` for that tool call.

2. **Partial tool calls are raw only.** A client that wants to continue a
partial tool call sends the raw tokens in a new `tool_calls_raw` string field
on the trailing assistant message. Structured `tool_calls` is allowed only for
**complete** tool calls (e.g. in-context learning examples).

3. **Tool calls live at the end.** When a prefill contains tool tokens, they
are appended after any reasoning/content in `generation_prompt_for_parser`.
If `tool_calls_raw` (or serialized complete `tool_calls`) is present, the
`content` field is considered closed.

4. **Reuse the PEG parser for continuation.** `generation_prompt_for_parser`
already contains the prefill text. The parser sees
`prefill_text + generated_text`, so tool-call completion works the same way as
cases 1–4 did for reasoning/content.

## What is already done

- Cases 1–4 prefill parsing works (`__prefill_is_continuation`,
`generation_prompt_for_parser` construction).
- `common_chat_params` carries tool-format metadata.
- `common_chat_tool_calls_to_text()` can serialize complete structured tool
  calls to raw model-format text.

## What is missing

1. Raw-token capture during PEG parsing.
2. `__raw` field in streaming/non-streaming `tool_calls` output.
3. `tool_calls_raw` prefill field.
4. Complete structured `tool_calls` prefill support.
5. `return_prefill` handling for tool calls.

---

## Implementation

### 1. Capture raw tool-call tokens in the PEG mapper

**File: `common/chat.h`**

Add a `raw` member to `common_chat_tool_call`:

```cpp
struct common_chat_tool_call {
    std::string id;
    std::string name;
    std::string arguments;
    std::string raw;   // exact raw model-format text for this tool call
};
```

**File: `common/chat-peg-parser.cpp`**

In `common_chat_peg_mapper::map()`, append `node.text` to the current tool
call's `raw` field for every tool-related tag:

- `TOOL_OPEN`
- `TOOL_ID`
- `TOOL_NAME`
- `TOOL_ARGS`
- `TOOL_ARG_OPEN`
- `TOOL_ARG_NAME`
- `TOOL_ARG_VALUE` / `TOOL_ARG_STRING_VALUE`
- `TOOL_ARG_CLOSE`
- `TOOL_CLOSE`

Example change in `map()`:

```cpp
auto append_raw = [&](const common_chat_tool_call * tc, std::string_view text) {
    if (tc) {
        const_cast<common_chat_tool_call *>(tc)->raw += text;
    }
};

if (is_tool_open) {
    pending_tool_call = common_chat_tool_call();
    current_tool = &pending_tool_call.value();
    arg_count = 0;
    args_buffer.clear();
    closing_quote_pending = false;
}

if (current_tool) {
    if (is_tool_open || is_tool_id || is_tool_name || is_tool_args ||
        is_arg_open || is_arg_name || is_arg_value || is_arg_string_value ||
        is_arg_close || is_tool_close) {
        current_tool->raw += node.text;
    }
}
```

Because the mapper is also used for partial streaming parses, `raw` will
naturally accumulate the exact tokens generated so far for an incomplete tool
call, and will hold the full exact tokens once the tool call closes.

### 2. Expose `__raw` in responses

**File: `common/chat.h`** (optional)

Add `raw` to `common_chat_msg_diff` tool-call deltas if it makes the server
side cleaner. Otherwise carry it via the `common_chat_tool_call` already added
in step 1.

**File: `tools/server/server-task.cpp`**

Wherever `tool_calls` are serialized (non-streaming `to_json_oaicompat_chat`
and streaming `to_json_oaicompat_chat_stream`), add a `__raw` field to each
tool-call object containing `tc.raw`.

For streaming, each `tool_calls` delta carries the **cumulative** raw text for
that tool call. The client can diff consecutive `__raw` values if it needs
per-token deltas, or simply use the final value for continuation.

### 3. Accept `tool_calls_raw` in request messages

**File: `common/chat.h`**

Add a `tool_calls_raw` string to `common_chat_msg`:

```cpp
struct common_chat_msg {
    std::string role;
    std::string content;
    std::string reasoning_content;
    std::vector<common_chat_tool_call> tool_calls;
    std::string tool_calls_raw;   // raw model-format tool tokens for prefill
};
```

**File: `common/chat.cpp`**

In `common_chat_msgs_parse_oaicompat()`, read the optional `tool_calls_raw`
field from assistant messages (string). Leave `tool_calls` parsing unchanged
for the standard structured array.

### 4. Detect and handle tool-call prefill in the server

**File: `tools/server/server-common.cpp`**

After the existing reasoning/content prefill detection in
`oaicompat_chat_params_parse`:

```cpp
std::string prefill_tool_calls_raw;
bool has_tool_calls = false;
bool has_tool_calls_raw = false;

if (prefill_assistant_message) {
    const auto & last_json_msg = messages.back();
    const auto & last_msg      = inputs.messages.back();

    // 1) explicit raw tool tokens
    if (last_json_msg.contains("tool_calls_raw") &&
        !last_json_msg.at("tool_calls_raw").is_null()) {
        has_tool_calls_raw = true;
        prefill_tool_calls_raw = last_json_msg.at("tool_calls_raw").get<std::string>();
    }

    // 2) complete structured tool_calls -> serialize to raw tokens
    if (!has_tool_calls_raw && !last_msg.tool_calls.empty()) {
        std::string raw = common_chat_tool_calls_to_text(last_msg.tool_calls, chat_params);
        if (!raw.empty()) {
            has_tool_calls = true;
            prefill_tool_calls_raw = raw;
        }
    }
}
```

If `prefill_tool_calls_raw` is non-empty, append it to
`generation_prompt_for_parser` after reasoning/content:

```cpp
if (!prefill_tool_calls_raw.empty()) {
    // If there is an open reasoning block (case 1), close it first, because
    // tool tokens mark the end of reasoning/content and the start of tools.
    if (prefill_case == 1) {
        generation_prompt_for_parser += chat_params.thinking_end_tag;
    }
    generation_prompt_for_parser += prefill_tool_calls_raw;
}
```

Also set a new prefill case value (e.g. `5`) and pass the raw text to the
slot:

```cpp
if (!prefill_tool_calls_raw.empty()) {
    prefill_case = 5;
    llama_params["__prefill_tool_calls_raw"] = prefill_tool_calls_raw;
    llama_params["__prefill_has_tool_calls"] = true;
}
```

### 5. Pass tool format metadata to the server

The serializer needs `common_chat_params`/`common_chat_tool_format`. Pass the
tool-format fields through `llama_params`:

**File: `tools/server/server-common.cpp`** after
`common_chat_templates_apply`:

```cpp
llama_params["__tool_format_mode"]        = chat_params.tool_format_mode;
llama_params["__tool_section_start"]      = chat_params.tool_section_start;
llama_params["__tool_section_end"]        = chat_params.tool_section_end;
llama_params["__tool_per_call_start"]     = chat_params.tool_per_call_start;
llama_params["__tool_name_prefix"]        = chat_params.tool_name_prefix;
llama_params["__tool_name_suffix"]        = chat_params.tool_name_suffix;
llama_params["__tool_close"]              = chat_params.tool_close;
llama_params["__tool_arguments_open"]     = chat_params.tool_arguments_open;
llama_params["__tool_arguments_close"]    = chat_params.tool_arguments_close;
llama_params["__tool_argument_name_prefix"]   = chat_params.tool_argument_name_prefix;
llama_params["__tool_argument_name_suffix"]   = chat_params.tool_argument_name_suffix;
llama_params["__tool_argument_value_prefix"]  = chat_params.tool_argument_value_prefix;
llama_params["__tool_argument_value_suffix"]  = chat_params.tool_argument_value_suffix;
llama_params["__tool_call_id_in_arguments"]   = chat_params.tool_call_id_in_arguments;
llama_params["__tool_call_id_key"]            = chat_params.tool_call_id_key;
```

**File: `tools/server/server-schema.cpp`**

Add handlers for these `__tool_*` fields and store them in
`common_chat_parser_params` (or a new `common_chat_tool_format` member of it).

**File: `tools/server/server-task.h` / `server-context.cpp`**

Carry the tool-format struct and `prefill_tool_calls_raw` through the task
and slot so `server-task.cpp` can serialize structured tool calls for
prefill and emit `__raw`.

### 6. `return_prefill` semantics for tool calls

**Storage:** keep the original prefilled structured `tool_calls`
(`last_msg.tool_calls`) and the raw string `prefill_tool_calls_raw` in the
result state.

**Streaming:** `is_continuation = true` already excludes the prefill from
diffs. For `return_prefill = true`, emit the prefilled tool calls as the
first delta(s) before the generated diffs. For `return_prefill = false`, emit
only generated diffs.

**Non-streaming:**

- `return_prefill = true`:
  - If the client sent structured `tool_calls`, output the exact original
    structured tool calls first, then any additional tool calls generated by
    the model.
  - If the client sent `tool_calls_raw`, the parsed result already contains
    the full tool calls (prefill + generated). Output them as parsed; the
    `__raw` field contains the exact raw tokens.

- `return_prefill = false`:
  - If the client sent structured complete `tool_calls` and the model added
    more, drop the first N prefilled tool calls and keep the rest.
  - If the client sent `tool_calls_raw` (partial), the model completed it.
    The response should contain the **completed** tool call, because the
    client needs the final structured result. The client can compute the
    model's raw contribution by subtracting its own prefill from `__raw`.

Do **not** try to output a partial structured tool call that represents only
the model's addition — that is not well-defined.

---

## Files to edit

1. `common/chat.h` — add `raw` to `common_chat_tool_call`; add
   `tool_calls_raw` to `common_chat_msg`; expose tool-format struct for
   parser params.
2. `common/chat.cpp` — parse `tool_calls_raw` in
   `common_chat_msgs_parse_oaicompat`.
3. `common/chat-peg-parser.cpp` — accumulate `node.text` into `tc.raw`.
4. `tools/server/server-common.cpp` — detect `tool_calls`/`tool_calls_raw`
   prefill, serialize complete structured tool calls, append to
   `generation_prompt_for_parser`, pass tool-format fields through
   `llama_params`.
5. `tools/server/server-schema.cpp` — read `__tool_*` and
   `__prefill_tool_calls_raw` fields.
6. `tools/server/server-task.h` — store tool-format and prefill tool-call
   state in task/result structs.
7. `tools/server/server-task.cpp` — emit `__raw` in tool-call output;
   implement `return_prefill` handling for tool calls.
8. `tools/server/server-context.cpp` — propagate fields from task to slot.

---

## Testing

1. **Streaming tool call `__raw`:** stream a tool call and verify every
   `tool_calls` delta has a `__raw` string that, concatenated across deltas,
   matches the raw tool-call text shown in `__verbose.content`.

2. **Non-streaming `__raw`:** make a non-streaming request that produces a
   tool call and verify `message.tool_calls[0].__raw` matches the raw
   tool-call tokens.

3. **Complete structured `tool_calls` prefill:** send a trailing assistant
   message with a complete `tool_calls` array and `return_prefill: true`.
   The model should not regenerate the prefilled tool call.

4. **Partial `tool_calls_raw` prefill:** take `__raw` from a previous
   interrupted response, send it as `tool_calls_raw` in a new trailing
   assistant message, and verify the model completes the tool call correctly.

5. **`return_prefill: false` with tool-call prefill:** verify the response
   excludes the prefilled structured tool calls but includes any newly
   generated ones (and, for partial raw prefill, includes the completed tool
   call).

6. **`return_prefill: true` with tool-call prefill:** verify the response
   includes the exact prefilled structured tool calls plus any new ones.
