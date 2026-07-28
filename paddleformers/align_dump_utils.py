# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
GLM4.5-Air 精度对齐辅助工具 (PaddleFormers 侧)

功能:
  1. 三个环境变量分别控制不同维度的对齐/日志, 默认全部关闭, 不影响 PaddleFormers 原有逻辑:
       GLM_ALIGN_BIT_EXACT=1   打开逻辑级对齐路径 (dataclass 默认覆盖、_keep_in_fp32_modules
                                剔除 mlp.gate.weight、aoa gate.weight dtype=bfloat16 等)。
       GLM_ALIGN_LOG=1         打开持续性插桩打印 (weight grad 等, 每个 step 都可能触发)。
       GLM_ALIGN_DUMP_DATA=1   打开一次性 dump (输入数据 md5/shape、初始权重 md5/norm)。

  2. 集中收纳 PF/MG 跨框架对齐用的 trainer 级别 dump:
       dump_input_info        打印 input_ids/position_ids/attention_mask 的 md5/shape
                              (受 GLM_ALIGN_DUMP_DATA 控制)
       dump_inputs_dict_info  从 inputs dict 中提取常见字段并打印
                              (受 GLM_ALIGN_DUMP_DATA 控制)
       dump_initial_weights   打印初始权重 md5/norm, 含 transpose / MoE 专家权重处理
                              (受 GLM_ALIGN_DUMP_DATA 控制)
       dump_weight_grads      打印 weight grad md5/norm (与 MG 侧 trainers/base.py 对齐)
                              (受 GLM_ALIGN_LOG 控制)

调用方默认通过开关函数早返回, 因此调用点写法是无条件 dump_xxx(...), 关闭时全部 no-op。
"""

import hashlib
import os

import numpy as np
import paddle

# ==================== 环境变量开关 ====================


def is_bit_exact() -> bool:
    """逻辑级对齐开关 (默认关闭, 走 PaddleFormers 原有逻辑)。"""
    return os.environ.get("GLM_ALIGN_BIT_EXACT", "0") == "1"


def is_log_enabled() -> bool:
    """持续性插桩日志开关 (weight grad 等, 默认关闭)。"""
    return os.environ.get("GLM_ALIGN_LOG", "0") == "1"


def is_dump_data_enabled() -> bool:
    """一次性 dump 开关 (输入数据、初始权重 md5/norm, 默认关闭)。"""
    return os.environ.get("GLM_ALIGN_DUMP_DATA", "0") == "1"


# ==================== 内部工具 ====================


def _global_step() -> int:
    """从 TRAINER_GLOBAL_STEP 推导当前 step (与原始插桩约定一致, 起始 1)。"""
    return int(os.environ.get("TRAINER_GLOBAL_STEP", "0")) + 1


def _md5_of_tensor(t) -> str:
    if t is None:
        return "None"
    if isinstance(t, paddle.Tensor):
        return hashlib.md5(t.numpy().tobytes()).hexdigest()
    if hasattr(t, "tobytes"):
        return hashlib.md5(t.tobytes()).hexdigest()
    return "N/A"


# ==================== 输入数据打印 ====================

_DEFAULT_INPUT_FIELDS = ("input_ids", "position_ids", "attention_mask", "labels", "loss_mask")


def dump_input_info(
    input_ids=None,
    position_ids=None,
    attention_mask=None,
    labels=None,
    loss_mask=None,
    attn_mask_startend_row_indices=None,
    tag="Paddle 输入数据",
    once_attr_owner=None,
    once_attr_name=None,
    only_first_step=True,
):
    """
    打印一组输入数据 (位置参数式调用) 的 md5 + shape, 由 GLM_ALIGN_DUMP_DATA 控制总开关。

    Args:
        once_attr_owner / once_attr_name:
            若提供, 第一次调用时在 owner 上 setattr(name, True) 并打印, 之后跳过。
        only_first_step:
            为 True 时仅当 TRAINER_GLOBAL_STEP+1 == 1 才打印 (与原始插桩一致)。
            若使用 once_attr_* 进行去重, 可设为 False。
    """
    if not is_dump_data_enabled():
        return
    if once_attr_owner is not None and once_attr_name is not None:
        if getattr(once_attr_owner, once_attr_name, False):
            return
    if only_first_step and _global_step() != 1:
        return

    print("\n" + "=" * 20 + f" [{tag}] " + "=" * 20)
    fields = [
        ("input_ids", input_ids),
        ("position_ids", position_ids),
        ("attention_mask", attention_mask),
        ("labels", labels),
        ("loss_mask", loss_mask),
    ]
    for name, t in fields:
        if t is None:
            continue
        shape_str = list(t.shape) if hasattr(t, "shape") else "N/A"
        print(f"Paddle {name} md5: {_md5_of_tensor(t)}, shape: {shape_str}")
    if attn_mask_startend_row_indices is not None:
        print(f"Paddle attn_mask_startend_row_indices md5: " f"{_md5_of_tensor(attn_mask_startend_row_indices)}")
    print("=" * 59 + "\n")

    if once_attr_owner is not None and once_attr_name is not None:
        setattr(once_attr_owner, once_attr_name, True)


def dump_inputs_dict_info(
    inputs,
    tag="Paddle 输入数据",
    only_first_step=True,
    once_attr_owner=None,
    once_attr_name=None,
    fields=_DEFAULT_INPUT_FIELDS,
):
    """从 inputs dict 中提取常见字段并打印 md5/shape。"""
    if not is_dump_data_enabled():
        return
    if once_attr_owner is not None and once_attr_name is not None:
        if getattr(once_attr_owner, once_attr_name, False):
            return
    if only_first_step and _global_step() != 1:
        return

    print("\n" + "=" * 5 + f" [{tag}] " + "=" * 5)
    for key in fields:
        t = inputs.get(key, None) if isinstance(inputs, dict) else None
        if isinstance(t, paddle.Tensor):
            print(f"Paddle {key} md5: {_md5_of_tensor(t)}, shape: {list(t.shape)}")
        elif t is None:
            print(f"Paddle {key}: None or Not Tensor")
        else:
            print(f"Paddle {key}: Not Tensor (type={type(t).__name__})")
    print("=" * 50 + "\n")

    if once_attr_owner is not None and once_attr_name is not None:
        setattr(once_attr_owner, once_attr_name, True)


# ==================== 初始权重打印 ====================

# 默认需要打印 transpose 版的权重名 (与 PF/MG 跨框架对齐时, 部分 weight 转置存储)
_DEFAULT_TRANSPOSE_WEIGHTS = (
    "self_attn.o_proj.weight",
    "self_attn.qkv_proj.weight",
    "mlp.up_gate_proj.weight",
    "mlp.down_proj.weight",
    "mlp.shared_experts.up_gate_proj.weight",
    "mlp.shared_experts.down_proj.weight",
    "experts.",  # expert 的 up_gate_proj 和 down_proj
)

# MoE grouped-gemm 专家权重: key 是 PF 参数名片段, value 是 MG 侧对应 fc 名
_DEFAULT_MOE_EXPERT_KEYS = {
    "grouped_gemm_experts.weight1": "linear_fc1",
    "grouped_gemm_experts.weight2": "linear_fc2",
}


def dump_initial_weights(
    model, tag="Paddle 初始权重", only_first_step=True, transpose_weights=None, moe_expert_keys=None, num_experts=128
):
    """打印模型所有参数的 md5/norm; 命中 transpose 列表的额外打印 transpose 版本; MoE 专家权重逐 expert 打印。"""
    if not is_dump_data_enabled():
        return
    if only_first_step and _global_step() != 1:
        return

    if transpose_weights is None:
        transpose_weights = _DEFAULT_TRANSPOSE_WEIGHTS
    if moe_expert_keys is None:
        moe_expert_keys = _DEFAULT_MOE_EXPERT_KEYS

    print("\n" + "=" * 20 + f" [{tag}] " + "=" * 20)
    for name, param in model.named_parameters():
        p = param.numpy()
        if p.dtype == np.uint16:
            p = param.astype("float32").numpy()
        else:
            p = p.astype(np.float32)
        md5 = hashlib.md5(p.tobytes()).hexdigest()
        norm_val = float(np.linalg.norm(p))
        print(f"[Paddle] {name} | md5: {md5} | norm: {norm_val:.6f} | " f"shape: {p.shape} | dtype: {param.dtype}")

        # transpose 版本
        if any(tw in name for tw in transpose_weights):
            p_transposed = p.T
            md5_t = hashlib.md5(p_transposed.tobytes()).hexdigest()
            norm_t = float(np.linalg.norm(p_transposed))
            print(
                f"[Paddle][Transposed] {name} | md5: {md5_t} | norm: {norm_t:.6f} | "
                f"shape: {p_transposed.shape} | dtype: {param.dtype}"
            )

        # MoE 专家权重逐 expert 打印 transpose 版
        for moe_key, fc_name in moe_expert_keys.items():
            if moe_key in name:
                for expert_id in range(num_experts):
                    expert_weight = p[expert_id].T
                    md5_e = hashlib.md5(expert_weight.tobytes()).hexdigest()
                    norm_e = float(np.linalg.norm(expert_weight))
                    print(
                        f"[Paddle[Transposed] ] _layers.2.mlp.experts.{fc_name}."
                        f"weight{expert_id} | md5: {md5_e} | norm: {norm_e:.6f} | "
                        f"shape: {expert_weight.shape} | dtype: {param.dtype}"
                    )
    print("=" * 61 + "\n")


# ==================== weight grad 打印 ====================


def _write_grad_info_pf(f, name, tensor):
    if tensor is None:
        f.write(f"| {'PF:'+name:<50s} | {'None':<16s} | {'N/A':<20s} | {'N/A':<12s} | {'N/A':<12s} |\n")
        return
    t = tensor.cast("float32")
    need_transpose = (
        t.ndim == 2
        and "embedding" not in name
        and "gate.weight" not in name
        and "layernorm" not in name
        and "norm" not in name
        and "_layers.4" not in name
        and "_layers.3" not in name
        and "bias" not in name
    )
    if need_transpose:
        data = t.t().contiguous().numpy()
    else:
        data = t.contiguous().numpy()
    md5 = hashlib.md5(data.tobytes()).hexdigest()[:16]
    shape_str = str(list(tensor.shape))
    dtype_str = str(tensor.dtype).replace("paddle.", "")
    norm_val = float(t.norm(p=2).item())
    f.write(f"| {'PF:'+name:<50s} | {md5:<16s} | {shape_str:<20s} | {dtype_str:<12s} | {norm_val:<12.6f} |\n")
    if "gate.weight" in name and tensor is not None:
        t_bf16 = tensor.cast("bfloat16").cast("float32")
        data_bf16 = t_bf16.contiguous().numpy()
        md5_bf16 = hashlib.md5(data_bf16.tobytes()).hexdigest()[:16]
        norm_bf16 = float(t_bf16.norm(p=2).item())
        f.write(
            f"| {'PF:'+name+'(bf16rt)':<50s} | {md5_bf16:<16s} | "
            f"{shape_str:<20s} | {'bf16->fp32':<12s} | {norm_bf16:<12.6f} |\n"
        )


def dump_weight_grads(model, only_first_step=True):
    """按 rank 写文件 pf_wgrad_rank{rank}.txt。受 GLM_ALIGN_LOG 控制。"""
    if not is_log_enabled():
        return
    cur_global_step = _global_step()
    if only_first_step and cur_global_step > 1:
        return
    rank_id = paddle.distributed.get_rank()
    out_dir = os.environ.get("WGRAD_DUMP_DIR", "/tmp")
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"pf_wgrad_rank{rank_id}.txt")
    with open(fpath, "w") as f:
        f.write(f"[PF] weight grad — step={cur_global_step} rank={rank_id}\n")
        f.write(f"| {'name':<50s} | {'md5(fp32)':<16s} | {'shape':<20s} | {'dtype':<12s} | {'norm':<12s} |\n")
        f.write(f"|{'-'*52}|{'-'*18}|{'-'*22}|{'-'*14}|{'-'*14}|\n")
        for name, param in model.named_parameters():
            grad = getattr(param, "main_grad", param.grad)
            _write_grad_info_pf(f, name, grad)
    if rank_id == 0:
        print(f"[ALIGN] weight grad written to {out_dir}/pf_wgrad_rank*.txt")


# ==================== Optimizer state probe (GLM_ALIGN_OPTIM_PROBE) ====================
# 与 GLM_ALIGN_LOG 解耦, 单独控制 optimizer 前/后状态对齐打印, 与 MG 侧
# mg_dump_optim_probe_print 输出格式一致, 便于跨框架 md5/norm 对比。
#
# Paddle 不像 Megatron 把 cast 拆成单独一步 (_copy_main_params_to_model_params),
# AdamW python 实现里 master_weight[:] = p ; param[:] = p.astype(param.dtype) 是
# 原子的, 所以这里只暴露 pre / post 两个 phase, 不再单独打 mid。
_optim_probe_step_pf = {"pf": 0}


def is_optim_probe_enabled() -> bool:
    """Optimizer 前后状态对齐打印开关 (默认关)"""
    return os.environ.get("GLM_ALIGN_OPTIM_PROBE", "0") == "1"


def _optim_probe_print_row_pf(label, arr):
    if arr is None:
        print(f"\033[35m[OPTIM PROBE] {label:<60s} None\033[0m", flush=True)
        return
    md5 = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    af32 = arr.astype(np.float32)
    norm = float(np.linalg.norm(af32))
    am = float(np.abs(af32).mean())
    mx = float(np.abs(af32).max())
    print(
        f"\033[35m[OPTIM PROBE] {label:<60s} md5={md5} shape={tuple(arr.shape)} "
        f"norm={norm:.6f} abs_mean={am:.4e} abs_max={mx:.4e}\033[0m",
        flush=True,
    )


def _pf_iter_optim_params(optimizer):
    """从 (可能经多层包装的) optimizer 里取出所有 param。
    顺序兼容: HybridParallelOptimizer -> MixPrecisionOptimizer -> AdamW。
    返回 (params_list, inner_adamw) ; inner_adamw 用于取 _master_weights / 累加器。
    """
    inner = optimizer
    # 穿透到底层 AdamW: MixPrecisionOptimizer / HybridParallelOptimizer 都暴露 _inner_opt
    for _ in range(4):
        if hasattr(inner, "_inner_opt") and getattr(inner, "_inner_opt") is not None:
            inner = inner._inner_opt
        else:
            break

    plist = getattr(optimizer, "_parameter_list", None)
    if plist is None:
        plist = getattr(inner, "_parameter_list", None)

    params = []
    if plist is not None and len(plist) > 0:
        if isinstance(plist[0], dict):
            for g in plist:
                params.extend(g.get("params", []))
        else:
            params = list(plist)
    elif hasattr(inner, "_param_groups"):
        for g in inner._param_groups:
            if isinstance(g, dict):
                params.extend(g.get("params", []))
            else:
                params.append(g)
    return params, inner


def pf_dump_optim_probe_print(phase, optimizer):
    """
    在 PF 侧 optimizer.step 不同 phase 调用一次, 紫色 [OPTIM PROBE] 行输出
    md5/shape/norm/abs_mean/abs_max, 与 MG 侧 mg_dump_optim_probe_print 同源。

    Args:
        phase: "pre" | "mid" | "post"
            pre  = optimizer.step() 前 (master fp32 + main_grad fp32 + moments)
            mid  = optimizer.step() 后, 用更新后的 moments 重算 update/denom (与 MG mid 对齐)
            post = master/working cast 后
        optimizer: trainer.self.optimizer (任意层包装均可)
    """
    if not is_optim_probe_enabled():
        return
    if phase == "pre":
        _optim_probe_step_pf["pf"] += 1
    n = _optim_probe_step_pf["pf"]

    try:
        params, inner = _pf_iter_optim_params(optimizer)
    except Exception as e:
        print(f"[OPTIM PROBE PF] iter params fail: {e}", flush=True)
        return

    master_weights = getattr(inner, "_master_weights", None) or {}
    moment1_str = getattr(inner, "_moment1_acc_str", "moment1")
    moment2_str = getattr(inner, "_moment2_acc_str", "moment2")
    get_acc = getattr(inner, "_get_accumulator_master", None)

    for pi, p in enumerate(params):
        if getattr(p, "stop_gradient", False):
            continue
        tag = f"p{pi}"
        pname = getattr(p, "name", None) or ""
        # 打印参数名便于跨框架映射
        if phase == "pre" and pi < 60:
            print(f"  [OPTIM PROBE PF] {tag} name={pname} shape={list(p.shape)}", flush=True)
        # 判断是否为 2D 权重需要转置来对齐 MG (Paddle: [out,in], MG: [in,out])
        _is_2d = p.ndim == 2
        # working param (原 dtype, 一般 bf16)
        try:
            wp_np = p.detach().contiguous().cast("float32").numpy()
            _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} working    ", wp_np)
            if _is_2d:
                _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} working.T  ", wp_np.T)
        except Exception as e:
            print(f"[OPTIM PROBE PF] {tag} working fail: {e}", flush=True)
        # master weight (fp32)
        mw = master_weights.get(pname) if pname else None
        if mw is not None:
            try:
                mw_np = mw.detach().contiguous().cast("float32").numpy()
                _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} master     ", mw_np)
                # master 是 flatten 的, 用 working shape reshape 后转置
                if _is_2d:
                    mw_2d = mw_np.reshape(p.shape)
                    _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} master.T   ", mw_2d.T)
            except Exception as e:
                print(f"[OPTIM PROBE PF] {tag} master fail: {e}", flush=True)
        # main_grad (fp32, 仅 pre 有意义)
        if phase == "pre" and hasattr(p, "main_grad") and p.main_grad is not None:
            try:
                g_np = p.main_grad.detach().contiguous().cast("float32").numpy()
                _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} main_grad  ", g_np)
                if _is_2d:
                    _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} main_grad.T", g_np.T)
            except Exception as e:
                print(f"[OPTIM PROBE PF] {tag} grad fail: {e}", flush=True)
        # === expert 参数逐 expert 切片 ===
        _is_moe_expert = "experts." in pname and ("weight1" in pname or "weight2" in pname) and p.ndim == 3
        if _is_moe_expert:
            num_local_experts = p.shape[0]
            if phase == "pre" and hasattr(p, "main_grad") and p.main_grad is not None:
                try:
                    g_full = p.main_grad.detach().contiguous().cast("float32").numpy()
                    for ei in range(num_local_experts):
                        g_i = g_full[ei]
                        _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} expert{ei} main_grad   ", g_i)
                        _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} expert{ei} main_grad.T ", g_i.T)
                except Exception as e:
                    print(f"[OPTIM PROBE PF] {tag} expert main_grad slice fail: {e}", flush=True)
            if mw is not None:
                try:
                    mw_full = mw.detach().contiguous().cast("float32").numpy()
                    mw_3d = mw_full.reshape(p.shape)
                    for ei in range(num_local_experts):
                        m_i = mw_3d[ei]
                        _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} expert{ei} master      ", m_i)
                        _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} expert{ei} master.T    ", m_i.T)
                except Exception as e:
                    print(f"[OPTIM PROBE PF] {tag} expert master slice fail: {e}", flush=True)
        # moments (Adam exp_avg / exp_avg_sq)
        if get_acc is not None:
            for sk_label, sk in (("exp_avg     ", moment1_str), ("exp_avg_sq  ", moment2_str)):
                try:
                    m = get_acc(sk, p)
                    if m is not None:
                        m_np = m.detach().contiguous().cast("float32").numpy()
                        _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} {sk_label}", m_np)
                        # moments 也是 flatten 的, reshape+T 输出
                        if _is_2d:
                            m_2d = m_np.reshape(p.shape)
                            _optim_probe_print_row_pf(f"pf step{n} {phase} {tag} {sk_label[:-5]}.T   ", m_2d.T)
                except Exception as e:
                    print(f"[OPTIM PROBE PF] {tag} {sk} fail: {e}", flush=True)
        # === Adam 中间量重算 (phase == "mid") ===
        if phase == "mid" and get_acc is not None and mw is not None:
            try:
                m1 = get_acc(moment1_str, p)
                m2 = get_acc(moment2_str, p)
                if m1 is not None and m2 is not None:
                    m1_np = m1.detach().contiguous().cast("float32").numpy()
                    m2_np = m2.detach().contiguous().cast("float32").numpy()
                    beta1 = float(getattr(inner, "_beta1", 0.9))
                    beta2 = float(getattr(inner, "_beta2", 0.95))
                    eps = float(getattr(inner, "_epsilon", 1e-8))
                    try:
                        lr = float(inner.get_lr())
                    except Exception:
                        lr = float(getattr(inner, "_learning_rate", 5e-5))
                        if callable(lr):
                            lr = 5e-5
                    step_t = float(n)
                    bc1 = 1.0 - beta1**step_t
                    bc2 = 1.0 - beta2**step_t
                    denom = np.sqrt(m2_np) / np.sqrt(bc2) + eps
                    update = lr * (m1_np / bc1) / denom
                    _optim_probe_print_row_pf(f"pf step{n} mid {tag} denom      ", denom)
                    _optim_probe_print_row_pf(f"pf step{n} mid {tag} update     ", update)
                    if _is_2d:
                        denom_2d = denom.reshape(p.shape)
                        update_2d = update.reshape(p.shape)
                        _optim_probe_print_row_pf(f"pf step{n} mid {tag} denom.T    ", denom_2d.T)
                        _optim_probe_print_row_pf(f"pf step{n} mid {tag} update.T   ", update_2d.T)
            except Exception as e:
                print(f"[OPTIM PROBE PF] {tag} adam mid recompute fail: {e}", flush=True)

    print(f"  [OPTIM PROBE PF] phase={phase} step{n} printed", flush=True)
