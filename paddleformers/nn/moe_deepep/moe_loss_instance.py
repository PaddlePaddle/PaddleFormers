

import paddle
from typing import Dict, Optional

from .moe_loss import LossCombiner, LossConfig, LossFunction, LossRegistry, LossType

# 全局损失注册器实例. 使用函数延迟创建实例
def get_global_loss_registry():
    if not hasattr(get_global_loss_registry, '_instance'):
        get_global_loss_registry._instance = LossRegistry()
        # 注册损失函数到全局注册器
        get_global_loss_registry._instance.register_loss("custom_diversity_loss1", custom_diversity_loss)
        # 注册combiner方法到全局注册器
        get_global_loss_registry._instance.register_combiner("custom_weighted_sum_combiner1", custom_weighted_sum_combiner)
    return get_global_loss_registry._instance


def custom_diversity_loss(
            routing_weights: paddle.Tensor,
            selected_experts: paddle.Tensor,
            gate_logits: Optional[paddle.Tensor] = None,
            **kwargs
    ) -> paddle.Tensor:
        """自定义多样性损失"""
        num_experts = kwargs.get('num_experts', 8)
        expert_counts = paddle.zeros([num_experts])

        for i in range(selected_experts.shape[0]):
            for j in range(selected_experts.shape[1]):
                expert_idx = selected_experts[i, j].item()
                expert_counts[expert_idx] += 1

        uniform_dist = paddle.ones_like(expert_counts) / expert_counts.shape[0]
        expert_probs = expert_counts / (expert_counts.sum() + 1e-8)

        diversity_loss = paddle.nn.functional.kl_div(
            paddle.log(expert_probs + 1e-8),
            paddle.log(uniform_dist + 1e-8),
            reduction='sum'
        )

        return diversity_loss

def custom_weighted_sum_combiner(
            self,
            losses: Dict[str, paddle.Tensor],
            configs: Dict[str, LossConfig]
    ) -> paddle.Tensor:
        """加权求和组合"""
        combined_loss = paddle.to_tensor(0.0)
        for name, loss_value in losses.items():
            config = configs.get(name)
            if config and config.enabled:
                combined_loss += config.weight * loss_value
        return combined_loss
        
