//
// moe-self-spec-test.cpp
//
// Self-speculative decoding test for DeepSeek MoE models.
// Drafts using shared expert only, verifies with full MoE.
// Measures acceptance rate, draft throughput, verify throughput, and effective speed.
//
// Build (from llama.cpp root, after applying moe-self-spec-complete.patch):
//   cmake --build build --config Release --target moe-self-spec-test
//
// Usage:
//   moe-self-spec-test -m model.gguf [-p prompt] [-n n_predict] [-d n_draft] [-c n_ctx] [-ngl n_gpu_layers]
//

#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>

using hrclock = std::chrono::high_resolution_clock;

static double elapsed_ms(hrclock::time_point t0, hrclock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// helper to add a token to a batch
static void batch_add(llama_batch & batch, llama_token id, llama_pos pos, llama_seq_id seq_id, bool logits) {
    int i = batch.n_tokens;
    batch.token[i]      = id;
    batch.pos[i]        = pos;
    batch.n_seq_id[i]   = 1;
    batch.seq_id[i][0]  = seq_id;
    batch.logits[i]     = logits ? 1 : 0;
    batch.n_tokens++;
}

// greedy sample from logits
static llama_token sample_greedy(llama_context * ctx, int idx) {
    float * logits = llama_get_logits_ith(ctx, idx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    llama_token best = 0;
    float best_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best = i;
        }
    }
    return best;
}

struct spec_stats {
    int total_draft    = 0;
    int total_accepted = 0;
    int total_tokens   = 0;  // total tokens produced
    double draft_ms    = 0;
    double verify_ms   = 0;

    double acceptance_rate() const {
        return total_draft > 0 ? (double)total_accepted / total_draft : 0.0;
    }

    double effective_tok_s() const {
        double total_ms = draft_ms + verify_ms;
        return total_ms > 0 ? (total_tokens / total_ms) * 1000.0 : 0.0;
    }
};

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  -m  <path>   model path (required)\n");
    fprintf(stderr, "  -p  <text>   prompt (default: \"Write a short story about a robot.\")\n");
    fprintf(stderr, "  -n  <int>    tokens to generate (default: 128)\n");
    fprintf(stderr, "  -d  <int>    draft tokens per step (default: 5)\n");
    fprintf(stderr, "  -e  <int>    draft expert count: 0=shared only, 1=top-1, 2=top-2, etc. (default: 0)\n");
    fprintf(stderr, "  -c  <int>    context size (default: 512)\n");
    fprintf(stderr, "  -ngl <int>   GPU layers to offload (default: 999)\n");
    fprintf(stderr, "  --baseline   also run baseline (no speculation) for comparison\n");
}

int main(int argc, char ** argv) {
    // defaults
    const char * model_path = nullptr;
    std::string  prompt     = "Write a short story about a robot.";
    int n_predict  = 128;
    int n_draft    = 5;
    int n_ctx      = 512;
    int n_gpu      = 999;
    int draft_n_expert = 0;  // 0=shared only, 1=top-1, etc.
    bool baseline  = false;

    // parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            n_draft = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            n_gpu = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            draft_n_expert = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--baseline") == 0) {
            baseline = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: model path required (-m)\n");
        print_usage(argv[0]);
        return 1;
    }

    // --- init model ---
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;

    fprintf(stderr, "Loading model: %s\n", model_path);
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // --- tokenize prompt ---
    std::vector<llama_token> prompt_tokens(prompt.size() + 32);
    int n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                  prompt_tokens.data(), prompt_tokens.size(),
                                  true, true);
    if (n_prompt < 0) {
        fprintf(stderr, "Tokenization failed\n");
        llama_model_free(model);
        return 1;
    }
    prompt_tokens.resize(n_prompt);

    fprintf(stderr, "Prompt tokens: %d\n", n_prompt);
    fprintf(stderr, "Draft length:  %d\n", n_draft);
    fprintf(stderr, "Draft experts: %d (0=shared only)\n", draft_n_expert);
    fprintf(stderr, "Generate:      %d tokens\n\n", n_predict);

    // === SELF-SPECULATIVE DECODING ===
    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx   = n_ctx;
        cparams.n_batch = n_ctx;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "Failed to create context\n");
            llama_model_free(model);
            return 1;
        }

        // eval prompt (full MoE)
        {
            llama_batch batch = llama_batch_init(n_prompt, 0, 1);
            for (int i = 0; i < n_prompt; i++) {
                batch_add(batch, prompt_tokens[i], i, 0, false);
            }
            batch.logits[n_prompt - 1] = true;

            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "Prompt eval failed\n");
                llama_batch_free(batch);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            llama_batch_free(batch);
        }

        // first token from prompt
        llama_token cur_token = sample_greedy(ctx, n_prompt - 1);
        int n_past = n_prompt;
        int n_generated = 0;

        spec_stats stats;

        fprintf(stderr, "=== Self-Speculative Decoding ===\n");
        fprintf(stderr, "Generating: ");

        auto t_total_start = hrclock::now();

        while (n_generated < n_predict) {
            // --- DRAFT PHASE: shared expert only ---
            llama_set_moe_draft_mode(ctx, draft_n_expert);

            std::vector<llama_token> draft_tokens;
            draft_tokens.push_back(cur_token);  // the "accepted" token from last round

            auto t_draft_start = hrclock::now();

            for (int d = 0; d < n_draft; d++) {
                llama_batch batch = llama_batch_init(1, 0, 1);
                batch_add(batch, draft_tokens.back(), n_past + d, 0, true);

                if (llama_decode(ctx, batch) != 0) {
                    llama_batch_free(batch);
                    break;
                }

                llama_token t = sample_greedy(ctx, 0);
                llama_batch_free(batch);
                draft_tokens.push_back(t);

                if (llama_vocab_is_eog(vocab, t)) break;
            }

            auto t_draft_end = hrclock::now();
            stats.draft_ms += elapsed_ms(t_draft_start, t_draft_end);

            // draft_tokens[0] = cur_token (already "accepted")
            // draft_tokens[1..] = speculative tokens to verify
            int n_to_verify = (int)draft_tokens.size(); // includes cur_token

            // --- VERIFY PHASE: full MoE, all tokens in one batch ---
            llama_set_moe_draft_mode(ctx, -1);

            // we need to remove KV for the draft positions and re-eval with full MoE
            // remove KV entries from n_past onwards (the draft tokens used partial graph)
            llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);

            auto t_verify_start = hrclock::now();

            llama_batch batch = llama_batch_init(n_to_verify, 0, 1);
            for (int i = 0; i < n_to_verify; i++) {
                batch_add(batch, draft_tokens[i], n_past + i, 0, true);
            }

            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                break;
            }

            // check how many draft tokens match full MoE
            int n_accepted = 0;
            for (int i = 0; i < n_to_verify - 1; i++) {
                llama_token verify_token = sample_greedy(ctx, i);
                if (verify_token == draft_tokens[i + 1]) {
                    n_accepted++;
                } else {
                    // mismatch: accept the verify token instead, stop here
                    cur_token = verify_token;
                    break;
                }
            }

            if (n_accepted == n_to_verify - 1) {
                // all draft tokens accepted, sample one more from the last position
                cur_token = sample_greedy(ctx, n_to_verify - 1);
                n_accepted++; // count the bonus token
            }

            auto t_verify_end = hrclock::now();
            stats.verify_ms += elapsed_ms(t_verify_start, t_verify_end);

            llama_batch_free(batch);

            // n_accepted tokens were added (draft_tokens[1..n_accepted] or up to n_to_verify)
            // plus cur_token for the next round
            int tokens_this_round;
            if (n_accepted == n_to_verify) {
                // all accepted + bonus
                tokens_this_round = n_to_verify;  // includes bonus
            } else {
                // n_accepted draft matched + 1 verify replacement
                tokens_this_round = n_accepted + 1;
            }

            // trim KV to only keep accepted tokens
            llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past + tokens_this_round, -1);
            n_past += tokens_this_round;
            n_generated += tokens_this_round;

            stats.total_draft += (n_to_verify - 1); // speculative tokens (excluding cur_token)
            stats.total_accepted += n_accepted;
            stats.total_tokens += tokens_this_round;

            // print accepted tokens
            for (int i = 1; i <= std::min(n_accepted, n_to_verify - 1); i++) {
                char buf[128];
                int len = llama_token_to_piece(vocab, draft_tokens[i], buf, sizeof(buf), 0, true);
                if (len > 0) fprintf(stderr, "%.*s", len, buf);
            }
            // print the verify/bonus token
            {
                char buf[128];
                int len = llama_token_to_piece(vocab, cur_token, buf, sizeof(buf), 0, true);
                if (len > 0) fprintf(stderr, "%.*s", len, buf);
            }

            if (llama_vocab_is_eog(vocab, cur_token)) break;
        }

        auto t_total_end = hrclock::now();
        double total_ms = elapsed_ms(t_total_start, t_total_end);

        fprintf(stderr, "\n\n=== Self-Speculative Results ===\n");
        fprintf(stderr, "Draft tokens:     %d\n", stats.total_draft);
        fprintf(stderr, "Accepted:         %d\n", stats.total_accepted);
        fprintf(stderr, "Acceptance rate:  %.1f%%\n", stats.acceptance_rate() * 100.0);
        fprintf(stderr, "Tokens generated: %d\n", stats.total_tokens);
        fprintf(stderr, "Draft time:       %.1f ms (%.1f t/s)\n", stats.draft_ms,
                stats.total_draft > 0 ? stats.total_draft / stats.draft_ms * 1000.0 : 0);
        fprintf(stderr, "Verify time:      %.1f ms\n", stats.verify_ms);
        fprintf(stderr, "Total time:       %.1f ms\n", total_ms);
        fprintf(stderr, "Effective speed:  %.1f t/s\n", stats.total_tokens / total_ms * 1000.0);

        llama_free(ctx);
    }

    // === BASELINE (optional) ===
    if (baseline) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx   = n_ctx;
        cparams.n_batch = n_ctx;

        llama_context * ctx = llama_init_from_model(model, cparams);

        // eval prompt
        {
            llama_batch batch = llama_batch_init(n_prompt, 0, 1);
            for (int i = 0; i < n_prompt; i++) {
                batch_add(batch, prompt_tokens[i], i, 0, false);
            }
            batch.logits[n_prompt - 1] = true;
            llama_decode(ctx, batch);
            llama_batch_free(batch);
        }

        llama_token cur_token = sample_greedy(ctx, n_prompt - 1);
        int n_past = n_prompt;
        int n_generated = 0;

        fprintf(stderr, "\n=== Baseline (no speculation) ===\n");
        fprintf(stderr, "Generating: ");

        auto t_start = hrclock::now();

        while (n_generated < n_predict) {
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch_add(batch, cur_token, n_past, 0, true);
            llama_decode(ctx, batch);
            llama_batch_free(batch);

            cur_token = sample_greedy(ctx, 0);
            n_past++;
            n_generated++;

            char buf[128];
            int len = llama_token_to_piece(vocab, cur_token, buf, sizeof(buf), 0, true);
            if (len > 0) fprintf(stderr, "%.*s", len, buf);

            if (llama_vocab_is_eog(vocab, cur_token)) break;
        }

        auto t_end = hrclock::now();
        double ms = elapsed_ms(t_start, t_end);

        fprintf(stderr, "\n\nBaseline speed: %.1f t/s (%d tokens in %.0f ms)\n",
                n_generated / ms * 1000.0, n_generated, ms);

        llama_free(ctx);
    }

    llama_model_free(model);

    return 0;
}
