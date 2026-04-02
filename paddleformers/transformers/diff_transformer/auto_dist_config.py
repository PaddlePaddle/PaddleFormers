AUTO_ATTN_IMPL = {"flashmask": "flashmask", "eager": "eager"}
AUTO_PARALLEL_STRATEGY = {"tp": 1, "pp": 1, "sp": False}

def get_auto_dist_config(config):
    return {
        "attn_implementation": AUTO_ATTN_IMPL.get(config._attn_implementation, "eager"),
        "tensor_parallel_degree": AUTO_PARALLEL_STRATEGY["tp"],
        "pipeline_parallel_degree": AUTO_PARALLEL_STRATEGY["pp"],
        "sequence_parallel": AUTO_PARALLEL_STRATEGY["sp"],
    }