"""Paddle modular_Qwen2 model."""
from fix_import import patcher
import math
import numpy as np
from paddlenlp.transformers import AutoModelForCausalLM
import paddle
from ...activations import ACT2FN
from ...contrastive_loss import SimpleContrastiveLoss
from ...embedding_utils import dist_gather_tensor_with_gradient
import os
from ..llama import parallel_matmul as llama_parallel_matmul
from ..llama import fusion_ops
from ...llama.modeling import get_use_casual_mask
from ..model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
import paddle.distributed as dist
import paddle.nn as nn
from paddle import DataParallel
from paddlenlp.transformers.llama.modeling import (
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    LlamaAttention,
    LlamaDecoderLayer
)
__all__=[
    "Qwen2Model",
    "Qwen2PretrainedModel",
    "Qwen2ForCausalLM",
    "Qwen2PretrainingCriterion",
    "Qwen2ForSequenceClassification",
    "Qwen2ForTokenClassification",
    "Qwen2SentenceEmbedding", 
]
from typing import Optional, Tuple, Union, List

def parallel_matmul(x, y, transpose_y=True, **kwargs):
    return llama_parallel_matmul(x, y, transpose_y=transpose_y, **kwargs)
# 旋转位置编码辅助函数

class Qwen2Config(LlamaConfig):
    """
    Qwen2模型配置类，基于PaddleNLP的LlamaConfig进行修改
    添加了Qwen2特有的配置参数
    """
    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # 默认使用GQA
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151643,
        tie_word_embeddings=False,
        rope_theta=1000000.0,  # Qwen2使用更大的rope_theta
        # Qwen2-specific additions
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_bias=True,
        attention_dropout=0.0,
        rope_scaling_factor=1.0,
        rope_scaling_type=None,
        use_fused_head_and_loss_fn=False,
        use_flash_attention=True,  # 添加Flash Attention支持
        use_dynamic_ntk=True,     # 动态NTK-aware插值
        logn_attn_scale=1.0,      # logn注意力缩放
        **kwargs
    ):
        # Set Qwen2-specific fields
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_fused_head_and_loss_fn = use_fused_head_and_loss_fn
        self.use_flash_attention = use_flash_attention
        self.use_dynamic_ntk = use_dynamic_ntk
        self.logn_attn_scale = logn_attn_scale

        # Rope & scaling
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_scaling_type = rope_scaling_type

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
"""
class Qwen2RMSNorm(nn.Layer):
    
    #Qwen2使用的RMSNorm实现

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[hidden_size],
            dtype='float32',
            default_initializer=nn.initializer.Constant(1.0)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
"""

class Qwen2Attention(LlamaAttention):
    """
    Qwen2注意力机制，基于PaddleNLP的LlamaAttention修改
    添加了GQA、动态NTK、Flash Attention等特性
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # GQA配置
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        """
        # 初始化query和key的RMSNorm
        self.q_norm = Qwen2RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )
        self.k_norm = Qwen2RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )
        """
        # Flash Attention支持
        self.use_flash_attention = (
            config.use_flash_attention and
            paddle.is_compiled_with_cuda() and
            hasattr(paddle.incubate.nn.functional, 'flash_attention')
        )

    def _compute_rope_freqs(self, seq_len):
        """计算旋转位置频率，支持动态NTK"""
        if self.config.use_dynamic_ntk:
            scale = (seq_len / 4096) ** (self.head_dim / (self.head_dim - 2))
            base = self.rope_theta * scale
        else:
            base = self.rope_theta
            
        freqs = 1.0 / (base ** (paddle.arange(0, self.head_dim, 2).astype("float32") / self.head_dim))
        return freqs

    def _apply_logn_attn(self, attention_scores, seq_len):
        """应用logn注意力缩放"""
        if seq_len <= 4096 or not hasattr(self.config, 'logn_attn_scale'):
            return attention_scores
            
        logn_scale = paddle.arange(1, seq_len + 1, dtype='float32').log() / math.log(seq_len)
        logn_scale = logn_scale * self.config.logn_attn_scale
        logn_scale = logn_scale.reshape([1, 1, seq_len, 1])
        return attention_scores * logn_scale

    def _apply_sliding_window_mask(self, attention_scores, seq_len):
        """应用滑动窗口注意力掩码"""
        if not self.config.use_sliding_window:
            return attention_scores
            
        window_size = self.config.sliding_window
        diagonal_mask = paddle.ones_like(attention_scores)
        row_indices = paddle.arange(seq_len).unsqueeze(1)
        col_indices = paddle.arange(seq_len).unsqueeze(0)
        mask = (col_indices - row_indices).abs() > window_size
        diagonal_mask = diagonal_mask.masked_fill(mask, -float('inf'))
        return attention_scores + diagonal_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影输入
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 应用RMSNorm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # 调整形状 [batch_size, seq_len, num_heads, head_dim]
        query_states = query_states.reshape(
            [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        key_states = key_states.reshape(
            [batch_size, seq_len, self.num_key_value_heads, self.head_dim]
        )
        value_states = value_states.reshape(
            [batch_size, seq_len, self.num_key_value_heads, self.head_dim]
        )
        
        # 计算旋转位置编码
        freqs = self._compute_rope_freqs(seq_len)
        cos, sin = self.rotary_emb(value_states, position_ids, freqs=freqs)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # GQA处理
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 使用Flash Attention
        if self.use_flash_attention and not output_attentions:
            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask
            )
            attn_weights = None
        else:
            # 常规注意力计算
            attn_weights = paddle.matmul(
                query_states, 
                key_states.transpose([0, 1, 3, 2])
            ) / math.sqrt(self.head_dim)
            
            # 应用logn缩放
            attn_weights = self._apply_logn_attn(attn_weights, seq_len)
            
            # 应用滑动窗口掩码
            attn_weights = self._apply_sliding_window_mask(attn_weights, seq_len)
            
            # 应用注意力掩码
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # 注意力概率和输出
            attn_weights = nn.functional.softmax(attn_weights, axis=-1)
            attn_output = paddle.matmul(attn_weights, value_states)
        
        # 输出处理
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, seq_len, -1])
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(self, query, key, value, attention_mask=None):
        """Flash Attention前向计算"""
        batch_size = query.shape[0]
        q = query.transpose([0, 2, 1, 3])  # [bs, nh, seq_len, hd]
        k = key.transpose([0, 2, 1, 3])    # [bs, nh, seq_len, hd]
        v = value.transpose([0, 2, 1, 3])  # [bs, nh, seq_len, hd]
        
        q, k, v = [x.reshape([batch_size * self.num_heads, x.shape[2], x.shape[3]]) for x in (q, k, v)]
        
        with paddle.fluid.dygraph.no_grad():
            output = paddle.incubate.nn.functional.flash_attention(
                q, k, v,
                dropout=self.config.attention_dropout,
                causal=True,
                return_softmax=False
            )
        
        output = output.reshape([batch_size, self.num_heads, output.shape[1], output.shape[2]])
        return output.transpose([0, 2, 1, 3])

class Qwen2MLP(nn.Layer):
    """
    Qwen2的MLP层，使用SiLU激活函数
    """
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=False)
        self.act_fn = nn.Swish()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen2DecoderLayer(LlamaDecoderLayer):
    """
    Qwen2解码器层，基于PaddleNLP的LlamaDecoderLayer修改
    使用Qwen2Attention和Qwen2MLP
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # 替换为Qwen2Attention
        self.self_attn = Qwen2Attention(config, layer_idx)
        
        # 使用Qwen2MLP
        self.mlp = Qwen2MLP(config)
        """
        # 使用RMSNorm
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        """
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        # Pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # 自注意力
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = residual + hidden_states
        
        # 前馈网络
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs

class Qwen2Model(LlamaModel):
    """
    Qwen2基础模型，基于PaddleNLP的LlamaModel修改
    """
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        
        # 替换解码器层
        self.layers = nn.LayerList(
            [Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        
        # 使用RMSNorm,直接用父类的RMSNorm
        #self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 初始化权重
        #self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # 复用父类forward逻辑
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

class Qwen2ForCausalLM(LlamaForCausalLM):
    """
    Qwen2因果语言模型，基于PaddleNLP的LlamaForCausalLM修改
    """
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        
        # 替换基础模型
        self.model = Qwen2Model(config)
        
        # 初始化权重

        #self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # 复用父类forward逻辑
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

if __name__ == "__main__":
    from paddlenlp.transformers import AutoTokenizer

    # 初始化配置
    config = Qwen2Config(
        vocab_size=151936,
        hidden_size=1024,
        num_hidden_layers=2,  # 测试时使用较少的层数
        num_attention_heads=8,
        num_key_value_heads=4,  # 测试GQA
        intermediate_size=11008,
        use_flash_attention=True,
        use_dynamic_ntk=True
    )

    # 创建模型
    # 创建模型并设置为评估模式
    model = Qwen2ForCausalLM(config)
    model.eval()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-7b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备输入
    input_text = "你好，请问"
    inputs = tokenizer(
        input_text,
        return_tensors="pd",
        max_length=32,
        truncation=True,
        padding="max_length"
    )

    # 生成文本
    with paddle.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 改进的输出处理（兼容tuple和tensor）
    def process_outputs(outputs):
        if isinstance(outputs, (tuple, list)):
            # 如果是tuple/list，取第一个元素（通常是生成的token ids）
            outputs = outputs[0]
        
        if isinstance(outputs, paddle.Tensor):
            outputs = outputs.numpy()
        
        if isinstance(outputs, np.ndarray):
            if len(outputs.shape) > 1:
                outputs = outputs[0]  # 取batch中的第一个样本
        
        return outputs.tolist() if hasattr(outputs, 'tolist') else outputs

    # 处理输出
    processed_output = process_outputs(outputs)
    
    # 安全解码
    try:
        decoded_text = tokenizer.decode(
            processed_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        print("生成结果:", decoded_text.strip())
    except Exception as e:
        print(f"解码失败: {e}")
        print("原始输出类型:", type(outputs))
        print("输出内容:", outputs)