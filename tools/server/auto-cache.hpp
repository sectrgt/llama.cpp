// auto-cache.hpp — Auto disk prompt/KV cache (opt-in: --slot-save-auto)
// Backported from Lissanro/llama.cpp prefill branch into upstream b12177.
#pragma once

#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <filesystem>
#include <chrono>
#include <atomic>
#include <algorithm>

// Magic/version for .meta sidecar files
static constexpr uint32_t SLOT_META_MAGIC   = 0x4D455441; // 'META'
static constexpr uint32_t SLOT_META_VERSION = 1;

// Model/quant/context fingerprint that MUST match for a restore to be sound. All
// fields are stable inference-affecting identity captured once at model load and
// compared by exact equality (pure-CPU int compares). See invariant 3. The blob
// produced by llama_state_seq_save_file is only safe to load into a context with
// identical KV geometry — a Q4_0-KV blob loaded into an F16 ctx, or a different
// rope/yarn scale (positions are baked into the saved state), silently corrupts —
// so cache_type_k/v and rope_scale are NOT optional.
struct model_fp {
    uint64_t fp_model      = 0; // hash of llama_model_desc + size + n_params (+ n_embd/n_layer)
    uint32_t fp_n_vocab    = 0;
    uint32_t fp_n_ctx_train= 0;
    uint32_t fp_n_embd     = 0;
    uint32_t fp_n_layer    = 0;
    uint32_t fp_rope_type  = 0;
    uint32_t fp_cache_k    = 0; // ggml_type of K cache (enum int)
    uint32_t fp_cache_v    = 0; // ggml_type of V cache (enum int)
    uint32_t fp_n_ctx      = 0; // effective per-seq n_ctx
    uint32_t fp_kv_full    = 0; // 1 if COMMON_CONTEXT_SEQ_RM_TYPE_FULL else 0
    uint32_t fp_block      = 0; // slot_save_block this snapshot was hashed with
    uint64_t fp_rope_scale = 0; // bit-pattern of effective rope_freq_scale (position-critical)
    // rope_freq_base and ALL YaRN params also bake positions into the saved KV state exactly as
    // rope_freq_scale does — a same-model run differing only in --rope-freq-base or any --yarn-*
    // flag would otherwise pass the fingerprint and silently restore positionally-corrupt state.
    // All are bit-cast (float->u32) into identity; yarn_orig_ctx is an int. "0/negative = use
    // model-trained value" is normalized in auto_compute_fingerprint so equal effective configs match.
    uint64_t fp_rope_base       = 0; // bit-pattern of effective rope_freq_base
    uint32_t fp_yarn_ext        = 0; // bit-pattern of yarn_ext_factor
    uint32_t fp_yarn_attn       = 0; // bit-pattern of yarn_attn_factor
    uint32_t fp_yarn_beta_fast  = 0; // bit-pattern of yarn_beta_fast
    uint32_t fp_yarn_beta_slow  = 0; // bit-pattern of yarn_beta_slow
    uint32_t fp_yarn_orig_ctx   = 0; // yarn_orig_ctx (int)
    uint64_t fp_lora       = 0; // hash of active LoRA-set ids+scales (0 if none)
    // refuse cross-shape restores: hash of the mmproj file path (0 if no mmproj).
    // The auto-cache only ever persists text-only prefixes, but different projectors
    // CAN alter the text KV layout (M-RoPE, different projection matrices, etc.),
    // so we REFUSE to cross-load snapshots from different projectors.
    uint64_t fp_mmproj_hash     = 0; // hash of mmproj path (0 if no mmproj)

    // exact field-by-field equality (C++17: no defaulted operator==). Any difference REFUSES the
    // restore (invariant 3). Note: fp_block is intentionally part of identity — a snapshot hashed
    // with a different block size cannot be longest-prefix-matched against the current index.
    // NOTE: fp_n_ctx is intentionally EXCLUDED from the comparison: n_ctx can vary slightly across
    // runs (e.g. padding/rounding) without invalidating the KV cache, and comparing it caused valid
    // caches to be rejected (the KV is valid as long as model arch + RoPE settings match).
    bool operator==(const model_fp & o) const {
        return fp_model == o.fp_model && fp_n_vocab == o.fp_n_vocab &&
               fp_n_ctx_train == o.fp_n_ctx_train && fp_n_embd == o.fp_n_embd &&
               fp_n_layer == o.fp_n_layer && fp_rope_type == o.fp_rope_type &&
               fp_cache_k == o.fp_cache_k && fp_cache_v == o.fp_cache_v &&
               fp_kv_full == o.fp_kv_full && fp_block == o.fp_block &&
               fp_rope_scale == o.fp_rope_scale && fp_rope_base == o.fp_rope_base &&
               fp_yarn_ext == o.fp_yarn_ext && fp_yarn_attn == o.fp_yarn_attn &&
               fp_yarn_beta_fast == o.fp_yarn_beta_fast && fp_yarn_beta_slow == o.fp_yarn_beta_slow &&
               fp_yarn_orig_ctx == o.fp_yarn_orig_ctx && fp_lora == o.fp_lora &&
               fp_mmproj_hash == o.fp_mmproj_hash;
    }
};

// 64-bit chained block hash over token IDs. Each token folds via FNV-1a then a
// splitmix avalanche; block k's output seeds block k+1, so the hash at every block
// boundary commits to the ENTIRE prefix [0, (k+1)*B). Collision resistance is only
// a candidate-narrowing accelerator: we NEVER trust it alone (invariant 2) — the
// caller byte-verifies tokens before any restore. Block boundaries are the only
// resumable prefix lengths (vLLM-APC / SGLang-radix granularity).
static inline uint64_t auto_hash_mix(uint64_t h, int32_t tok) {
    h ^= (uint64_t) (uint32_t) tok;
    h *= 0x100000001b3ULL;                                  // FNV-1a 64-bit prime
    h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 32; // splitmix64 finalize
    return h;
}

// FNV-1a hash of a NUL-terminated C string (used to fold media chunk IDs into the block hash).
// mtmd_input_chunk_get_id() returns the image/audio content hash (set by mtmd_helper_bitmap_init
// from_buf to the FNV-1a of the raw bytes), so two different images yield different chunk IDs and
// thus different block hashes — the core of media-aware cache key differentiation.
static inline uint64_t auto_chunk_id_hash(const char * id) {
    uint64_t h = 0xcbf29ce484222325ULL;
    if (id) {
        for (const char * p = id; *p; ++p) {
            h ^= (uint64_t) (unsigned char) *p;
            h *= 0x100000001b3ULL;
        }
    }
    return h;
}

// Extract a (chunk-start-index -> content-hash) map from a prompt's media chunks. Used by the
// auto disk cache so that two prompts with identical text tokens but different images produce
// different block hashes (and thus different cache entries). Returns an empty map for text-only
static std::map<size_t, uint64_t> auto_media_hashes(const server_tokens & tokens) {
    // Simplified: returns empty map (no media-aware hashing in initial backport)
    std::map<size_t, uint64_t> hashes;
    return hashes;
}

// Returns, for each block boundary b in [1 .. n/B], the cumulative chain hash
// committing to tokens[0 .. b*B). out[k] = hash of prefix length (k+1)*B. The
// chain is salted with `salt` (the model fingerprint hash) so two different models
// can never produce the same boundary hash for identical tokens. A trailing
// partial block is NOT a boundary (only whole-block prefixes are index keys).
//
// `media_hashes` maps a chunk's START token index to a content hash (auto_media_hashes).
// At each NULL (LLAMA_TOKEN_NULL) position that is a chunk start, the chunk hash is folded
// into the chain AFTER the token ID, so two prompts with the same text but different images
// produce different block hashes. Interior NULL positions (non-start) have no media hash entry
// and fold only the NULL token ID (identical for all images), preserving the chain position.
static std::vector<uint64_t> auto_block_hashes(const llama_tokens & toks, int B, uint64_t salt,
                                                const std::map<size_t, uint64_t> & media_hashes) {
    std::vector<uint64_t> out;
    if (B <= 0) {
        return out;
    }
    out.reserve(toks.size() / (size_t) B);
    uint64_t h = 0xcbf29ce484222325ULL ^ salt; // FNV offset basis, fingerprint-salted
    for (size_t i = 0; i < toks.size(); ++i) {
        h = auto_hash_mix(h, toks[i]);
        // at a media chunk start, fold the chunk's content hash so that two prompts with
        // identical text but different images produce different block hashes.
        if (toks[i] == LLAMA_TOKEN_NULL) {
            auto it = media_hashes.find(i);
            if (it != media_hashes.end()) {
                h = auto_hash_mix(h, (int32_t) (it->second & 0xFFFFFFFFu));
                h = auto_hash_mix(h, (int32_t) (it->second >> 32));
            }
        }
        if ((i + 1) % (size_t) B == 0) {
            out.push_back(h);
        }
    }
    return out;
}

struct auto_cache_entry {
    std::string state_path; // full state file path (sidecars derived via *_path helpers)
    uint32_t    n_tokens = 0;
    model_fp    fp;         // snapshot's fingerprint (must equal the live one to be used)
};

// boundary-hash -> best (longest) entry covering that prefix length. Touched only
// from the single server-loop thread in v1 (mtx documented above). `scanned`
// guards the one-time startup scan; `dir_mtime`/`last_refresh` drive the cheap
// cross-process refresh (see auto_index_refresh): a peer process that writes a new
struct auto_cache_index {
    std::mutex mtx;
    std::unordered_map<uint64_t, auto_cache_entry> by_boundary;
    std::unordered_set<std::string> indexed_files;    // state paths already scanned (incremental refresh)
    bool scanned = false;
    std::filesystem::file_time_type dir_mtime{};      // dir mtime as of the last scan
    std::chrono::steady_clock::time_point last_refresh{}; // throttle: skip stat storms in a burst
};

// Cross-process refresh throttle: at most one dir-mtime stat per this interval on the hot lookup
// path (a forced refresh on a lookup miss bypasses it). Sub-second so a peer's new snapshot is
// visible within ~1 prefill of being written — effectively immediate from the user's view.
static constexpr int AUTO_REFRESH_MIN_MS = 1000;

// Sidecar path twins for an auto snapshot's state file. `.logits` is the committed
// (byte-identical) regenerate sidecar; `.meta` is the NEW tokens+fingerprint
// sidecar this feature adds so the startup scan / pre-restore verify reads only a
static std::string slot_meta_sidecar_path(const std::string & state_filepath) {
    return state_filepath + ".meta";
}

// Best-effort atomic write of the .meta sidecar (LE, temp+rename — the exact idiom
// of slot_logits_write). Layout: magic/version, fingerprint fields, tok_count,
// chain_hash, int32 tokens[tok_count], then media_hashes (n_media u32 + per-entry
// position u64 + chunk_id_hash u64) so the startup scan can compute media-aware
// block hashes without reading the multi-GB .bin or the .stokens chunk data.
// Returns true on success. Never throws.
static bool slot_meta_write(const std::string & state_filepath,
                            const model_fp & fp,
                            const llama_tokens & toks,
                            uint64_t chain_hash,
                            const std::map<size_t, uint64_t> & media_hashes) {
    const std::string sidecar = slot_meta_sidecar_path(state_filepath);
    const std::string tmp     = sidecar + ".tmp";

    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    auto put_u32 = [&](uint32_t v) {
        const unsigned char b[4] = {
            (unsigned char)( v        & 0xFF),
            (unsigned char)((v >> 8)  & 0xFF),
            (unsigned char)((v >> 16) & 0xFF),
            (unsigned char)((v >> 24) & 0xFF),
        };
        f.write((const char *) b, 4);
    };
    auto put_u64 = [&](uint64_t v) {
        put_u32((uint32_t)(v & 0xFFFFFFFFu));
        put_u32((uint32_t)(v >> 32));
    };
    put_u32(SLOT_META_MAGIC);
    put_u32(SLOT_META_VERSION);
    put_u64(fp.fp_model);
    put_u32(fp.fp_n_vocab);
    put_u32(fp.fp_n_ctx_train);
    put_u32(fp.fp_n_embd);
    put_u32(fp.fp_n_layer);
    put_u32(fp.fp_rope_type);
    put_u32(fp.fp_cache_k);
    put_u32(fp.fp_cache_v);
    put_u32(fp.fp_n_ctx);
    put_u32(fp.fp_kv_full);
    put_u32(fp.fp_block);
    put_u64(fp.fp_rope_scale);
    // rope_freq_base + YaRN fingerprint fields
    put_u64(fp.fp_rope_base);
    put_u32(fp.fp_yarn_ext);
    put_u32(fp.fp_yarn_attn);
    put_u32(fp.fp_yarn_beta_fast);
    put_u32(fp.fp_yarn_beta_slow);
    put_u32(fp.fp_yarn_orig_ctx);
    put_u64(fp.fp_lora);
    // mmproj projector hash (u64): different projectors get disjoint stores.
    put_u64(fp.fp_mmproj_hash);
    put_u32((uint32_t) toks.size());
    put_u64(chain_hash);
    // token IDs as raw LE int32 (llama_token == int32_t; llama.cpp's on-disk
    // contract is native-LE, matching slot_logits_write's float payload).
    f.write((const char *) toks.data(), (std::streamsize) toks.size() * sizeof(int32_t));
    // media chunk content hashes: (start-index -> FNV of chunk ID) for each media chunk.
    // Lets the scan compute media-aware block hashes (same text + different image = different
    // key) without reading the .stokens sidecar. Text-only snapshots write an empty map.
    put_u32((uint32_t) media_hashes.size());
    for (const auto & [idx, hash] : media_hashes) {
        put_u64((uint64_t) idx);
        put_u64(hash);
    }
    f.flush();
    if (!f.good()) {
        f.close();
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        return false;
    }
    f.close();
    std::error_code ec;
    std::filesystem::rename(tmp, sidecar, ec); // atomic replace
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

