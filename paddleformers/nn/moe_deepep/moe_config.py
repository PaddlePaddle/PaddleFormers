MOE_CONFIG = {
    "qwen3_moe": {
        "gate_activation": "softmax",
        "expert_activation": "silu",
        "train_topk_method": "greedy",
        "inference_topk_method": "greedy",
        "aux_loss_weight": 0.01,
        "z_loss_weight": 0.0,
        "expert_dropout": 0.0,
        "use_flexible_loss": False,
        "moe_group": "expert",
        "drop_tokens": False,
        "custom_expert": "MLP"
    }
}