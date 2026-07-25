# llama.cpp tool-call continuation with prefill — implementation plan

## Problem

When `return_prefill: true` (or any prefill case 1–4) is active and the model
emits a tool call, the tool-call tokens are returned as raw text in `content`
with `finish_reason: "stop"` instead of being parsed into `message.tool_calls`
with `finish_reason: "tool_calls"`.

**Root cause** (commit `cefd143f`): `task_result_state::update_chat_msg()`
bypasses `common_chat_parse()` for prefill cases 1–4 and manually splits the
generated text into `reasoning_content` / `content`.  The PEG parser — which
extracts tool calls — is only invoked for case 0 (no prefill).  So tool-call
syntax in the generated text is never parsed.

## Key insight

The existing **continuation mechanism** (`continue_final_message`) already
handles tool calls correctly.  It sets `chat_parser_params.generation_prompt`
to the prefilled text and lets `common_chat_parse()` process
`generation_prompt + generated_text` with the PEG parser.  The prefill patch
replaced this with custom token-splicing for the **prompt** (which is fine)
but also bypassed the PEG parser for **parsing** (which breaks tool calls).

**Fix:** keep the token-splicing for prompt construction, but restore PEG-parser
parsing by setting `chat_parser_params.generation_prompt` to the prefill text
so `common_chat_parse()` sees `prefill_text + generated_text` and parses tool
calls normally.

## How the PEG parser + generation_prompt work

`common_chat_peg_parse()` prepends `generation_prompt` to the input:
```
effective_input = generation_prompt + input
```
The auto-parser builds a parser of the form:
```
prefix(parser_generation_prompt, thinking_start) + optional(reasoning_block) + content_or_tools
```
where `parser_generation_prompt` is the **original** `generation_prompt`
(baked into the parser at build time, before continuation modification) and
`prefix(s, delim) = literal(s.substr(0, s.find(delim)))`.

So `effective_input` must start with `prefix(parser_generation_prompt,
thinking_start)`, then optionally a reasoning block, then content/tools.

By setting `chat_parser_params.generation_prompt` to the prefill text, the
parser sees:
```
prefill_text + generated_text
= [assistant_prefix] [thinking_start] prefill_reasoning [thinking_end] prefill_content  +  generated_text
```
and correctly separates reasoning, content, **and tool calls**.

## Construction of `generation_prompt_for_parser` per case

Using `chat_params.generation_prompt` (the normal gen-prompt from the template),
`chat_params.thinking_start_tag` / `thinking_end_tag` (trimmed reasoning tags),
and the prefill content:

```
has_thinking  =  !thinking_start_tag.empty()
               && generation_prompt.find(thinking_start_tag) != npos
assistant_prefix = has_thinking
                 ? generation_prompt.substr(0, generation_prompt.find(thinking_start_tag))
                 : generation_prompt
```

| Case | generation_prompt_for_parser |
|------|------|
| 1 (reasoning only, open) | `generation_prompt + prefill_reasoning` (gen-prompt already includes `thinking_start`) |
| 2 (reasoning + content) | `generation_prompt + prefill_reasoning + thinking_end_tag + prefill_content` |
| 3 (content only) | `assistant_prefix + prefill_content` (strip `thinking_start` from gen-prompt) |
| 4 (reasoning + empty content) | `generation_prompt + prefill_reasoning + thinking_end_tag` |

When `!has_thinking` (model has no thinking or thinking disabled):
- Cases 1/2/4: prepend `thinking_start_tag` after `generation_prompt`
  (`generation_prompt + thinking_start_tag + prefill_reasoning + ...`)
- Case 3: `generation_prompt + prefill_content`

This mirrors exactly what the auto-parser's continuation code does in
`common_chat_auto_parser_generator.cpp` (lines 53-69), but covers all four
prefill cases including case 3 (content-only) which the auto-parser mishandles
by inserting empty thinking tags.

## `is_continuation` and `return_prefill`

Setting `is_continuation = true` causes `task_result_state`'s constructor to
initialize `chat_msg` by parsing `generation_prompt` (the prefill text) alone.
This means:
- diffs (streaming) **exclude** the prefill — correct for `return_prefill=false`
- `oaicompat_msg` (final) **includes** the prefill

| | `return_prefill=false` | `return_prefill=true` |
|---|---|---|
| **non-streaming** | subtract prefill from `oaicompat_msg` | keep as-is |
| **streaming** | diffs already correct | emit prefill as first delta |

## Files to modify

### 1. `tools/server/server-common.cpp` — `oaicompat_chat_params_parse()`

After `common_chat_templates_apply()` and after prefill detection, construct
`generation_prompt_for_parser` and override `llama_params["generation_prompt"]`.

Also pass `thinking_start_tag` / `thinking_end_tag` to the slot via
`llama_params` (they're already passed as `reasoning_budget_start_tag` /
`reasoning_budget_end_tag` but only when `thinking_end_tag` is non-empty —
pass them unconditionally for prefill cases).

Set `llama_params["__prefill_is_continuation"] = true` so the server sets
`is_continuation = true`.

### 2. `tools/server/server-schema.cpp`

Add a handler for `__prefill_is_continuation` that sets
`chat_parser_params.is_continuation = true`.

### 3. `tools/server/server-task.cpp` — `update_chat_msg()`

**Remove** the manual case 1–4 splitting.  For all cases (0–4), call
`common_chat_parse(generated_text, is_partial, chat_parser_params)`.

The `chat_parser_params.generation_prompt` is already set to the prefill text
(in `server-common.cpp`), so `common_chat_parse` processes
`prefill_text + generated_text` and correctly parses reasoning, content, and
tool calls.

### 4. `tools/server/server-task.cpp` — `return_prefill` handling

**`server_task_result_cmpl_final::update()`**:
- Remove the old prepend logic.
- When `!return_prefill && prefill_case > 0`: subtract `prefill_reasoning_content`
  from `oaicompat_msg.reasoning_content` and `prefill_content` from
  `oaicompat_msg.content` (the parser already includes them).
- When `return_prefill`: do nothing (parser output already includes prefill).

**`server_task_result_cmpl_partial::update()`**:
- Remove the old prepend-to-first-delta logic.
- When `return_prefill && prefill_case > 0 && n_decoded == 1`: emit prefill
  as a synthetic delta (reasoning and/or content) before the parsed diffs.

**`to_json_oaicompat_chat()`**:
- Remove the old prepend logic (the prefill is now in `oaicompat_msg` from the
  parser; subtraction for `!return_prefill` is done in `update()`).

### 5. `tools/server/server-task.h`

- Remove `prefill_opening_sequence` and `prefill_closing_sequence` from
  `task_result_state` and the result structs (no longer needed — the PEG parser
  handles reasoning/content separation).
- Keep `prefill_case`, `prefill_reasoning_content`, `prefill_content`,
  `return_prefill` (still needed for `return_prefill` subtraction/emission).
- Add `prefill_is_continuation` field (or reuse `is_continuation` from
  `chat_parser_params`).

### 6. `tools/server/server-context.cpp` — `build_prefill_tokens()`

- `prefill_opening_sequence` / `prefill_closing_sequence` are no longer needed
  for parsing (the PEG parser handles it).  They can be removed from the slot
  and result structs.
- The token-splicing logic stays unchanged (it builds the prompt tokens).

## Testing

1. **Case 3 (the reported bug):** reasoning prefill + tool call → should
   return structured `tool_calls` with `finish_reason: "tool_calls"`.
2. **Case 0 (control):** no prefill + tool call → should still work.
3. **Case 1:** reasoning-only prefill, model generates reasoning + tool call →
   tool call should be parsed.
4. **Case 2:** reasoning + content prefill, model generates tool call → tool
   call should be parsed.
5. **`return_prefill=false` (default):** response should NOT include prefilled
   text, only generated text.
6. **`return_prefill=true`:** response should include prefilled text + generated
   text.
7. **Streaming:** deltas should correctly include/exclude prefill based on
   `return_prefill`.
8. **Non-reasoning models:** case 3 prefill + tool call should work.
