// llama-stage-server: one earlier stage of a model split by layers across processes (common/stage-link.h).
//
// It loads its part of the model with the usual options and serves, on --stage-serve HOST:PORT, the hidden state its
// last layer hands on for every token the next stage sends. With --stage-prev it takes its own input from the stage
// before it, so a middle stage is the same program. The token both sides check is LLAMA_STAGE_TOKEN.
//
//   LLAMA_STAGE_TOKEN=... llama-stage-server -m slice-0.gguf -c 32768 -np 1 --stage-serve 127.0.0.1:9000

#include "arg.h"
#include "common.h"
#include "log.h"
#include "stage-link.h"

#include <cstdlib>

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    common_init();
    if (params.stage_serve.empty()) {
        LOG_ERR("--stage-serve HOST:PORT is required\n");
        return 1;
    }
    const char * token = getenv("LLAMA_STAGE_TOKEN");
    if (!token || !*token) {
        LOG_ERR("LLAMA_STAGE_TOKEN is required\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (!ctx) {
        LOG_ERR("failed to load the model\n");
        return 1;
    }
    const int rc = stage_link::serve(ctx, params.stage_serve, token, (int32_t) params.n_batch);
    llama_backend_free();
    return rc;
}
