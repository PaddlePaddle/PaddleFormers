# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.vision.models import resnext50_64x4d

from paddleformers.fleet.cudagraph import autocudagraph


def set_strict_seeds(seed=2026):
    paddle.seed(seed)
    np.random.seed(seed)
    paddle.set_flags(
        {
            "FLAGS_cudnn_deterministic": 1,
        }
    )


def assert_tensors_close(self, t1, t2, rtol=0, atol=0, msg=""):
    # Assumes 100% precision alignment (rtol=0, atol=0) because CUDAGraph does not alter the computational graph's execution order under deterministic settings.
    if t1 is None and t2 is None:
        return
    self.assertIsNotNone(t1, f"{msg}: t1 is None but t2 is not")
    self.assertIsNotNone(t2, f"{msg}: t2 is None but t1 is not")
    np.testing.assert_allclose(t1.numpy(), t2.numpy(), rtol=rtol, atol=atol, err_msg=msg)


class BaseTest(unittest.TestCase):
    def setUp(self):
        set_strict_seeds()
        paddle.device.cuda.empty_cache()


def pure_func_eager(x, weight, bias):
    return F.relu(F.linear(x, weight, bias))


@autocudagraph(warmup_steps=2)
def pure_func_cg(x, weight, bias):
    return F.relu(F.linear(x, weight, bias))


class TestPureFunctions(BaseTest):
    def tearDown(self):
        pure_func_cg.clear_cache()

    def test_multi_inputs_and_stop_gradient(self):
        x_cg = paddle.randn([4, 16])
        x_cg.stop_gradient = False
        w_cg = paddle.randn([16, 32])
        w_cg.stop_gradient = False
        b_cg = paddle.randn([32])
        b_cg.stop_gradient = True

        x_eager = x_cg.clone().detach()
        x_eager.stop_gradient = False
        w_eager = w_cg.clone().detach()
        w_eager.stop_gradient = False
        b_eager = b_cg.clone().detach()
        b_eager.stop_gradient = True

        for step in range(5):
            out_cg = pure_func_cg(x_cg, w_cg, b_cg)
            loss_cg = out_cg.mean()
            loss_cg.backward()

            out_eager = pure_func_eager(x_eager, w_eager, b_eager)
            loss_eager = out_eager.mean()
            loss_eager.backward()

            assert_tensors_close(self, loss_cg, loss_eager, msg=f"Step {step} Loss")

            assert_tensors_close(self, x_cg.grad, x_eager.grad, msg=f"Step {step} x.grad")
            assert_tensors_close(self, w_cg.grad, w_eager.grad, msg=f"Step {step} w.grad")

            self.assertIsNone(b_cg.grad)

            x_cg.clear_gradient()
            w_cg.clear_gradient()
            x_eager.clear_gradient()
            w_eager.clear_gradient()


class BaseImplicitLayer(nn.Layer):
    def __init__(self, dim=16):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward_logic(self, x):
        return self.linear2(F.relu(self.norm(self.linear1(x))))


class ImplicitLayerEager(BaseImplicitLayer):
    def forward(self, x):
        return self.forward_logic(x)


class ImplicitLayerCG(BaseImplicitLayer):
    @autocudagraph(warmup_steps=2)
    def forward(self, x):
        return self.forward_logic(x)


class TestOOPAndSubmodules(BaseTest):
    def tearDown(self):
        ImplicitLayerCG.forward.clear_cache()

    def test_pointer_drift_and_weight_update(self):
        model_cg = ImplicitLayerCG()
        model_eager = ImplicitLayerEager()
        model_eager.set_state_dict(model_cg.state_dict())

        opt_cg = paddle.optimizer.Adam(learning_rate=0.01, parameters=model_cg.parameters())
        opt_eager = paddle.optimizer.Adam(learning_rate=0.01, parameters=model_eager.parameters())

        for step in range(5):
            x = paddle.randn([8, 16])

            loss_cg = model_cg(x).sum()
            loss_cg.backward()
            opt_cg.step()
            opt_cg.clear_grad()

            loss_eager = model_eager(x).sum()
            loss_eager.backward()
            opt_eager.step()
            opt_eager.clear_grad()

            assert_tensors_close(self, loss_cg, loss_eager, msg=f"Step {step} OOP Loss")
            assert_tensors_close(
                self,
                model_cg.linear1.weight,
                model_eager.linear1.weight,
                msg=f"Step {step} linear1.weight pointer drift",
            )
            assert_tensors_close(
                self,
                model_cg.linear2.weight,
                model_eager.linear2.weight,
                msg=f"Step {step} linear2.weight pointer drift",
            )

    def test_multi_instances_isolation_with_eager(self):
        eager_obj1 = ImplicitLayerEager()
        eager_obj2 = ImplicitLayerEager()
        cg_obj1 = ImplicitLayerCG()
        cg_obj2 = ImplicitLayerCG()

        cg_obj1.set_state_dict(eager_obj1.state_dict())
        cg_obj2.set_state_dict(eager_obj2.state_dict())

        opt_eager1 = paddle.optimizer.Adam(learning_rate=0.01, parameters=eager_obj1.parameters())
        opt_cg1 = paddle.optimizer.Adam(learning_rate=0.01, parameters=cg_obj1.parameters())
        opt_eager2 = paddle.optimizer.Adam(learning_rate=0.01, parameters=eager_obj2.parameters())
        opt_cg2 = paddle.optimizer.Adam(learning_rate=0.01, parameters=cg_obj2.parameters())

        for step in range(5):
            x1 = paddle.randn([4, 16])
            x_eager1 = x1.clone().detach()
            x_eager1.stop_gradient = False
            x_cg1 = x1.clone().detach()
            x_cg1.stop_gradient = False

            x2 = paddle.randn([4, 16])
            x_eager2 = x2.clone().detach()
            x_eager2.stop_gradient = False
            x_cg2 = x2.clone().detach()
            x_cg2.stop_gradient = False

            out_eager1 = eager_obj1(x_eager1)
            out_cg1 = cg_obj1(x_cg1)
            out_eager2 = eager_obj2(x_eager2)
            out_cg2 = cg_obj2(x_cg2)

            assert_tensors_close(self, out_cg1, out_eager1, msg=f"Step {step} Obj1 Forward")
            assert_tensors_close(self, out_cg2, out_eager2, msg=f"Step {step} Obj2 Forward")

            out_eager1.mean().backward()
            out_cg1.mean().backward()
            out_eager2.mean().backward()
            out_cg2.mean().backward()

            assert_tensors_close(self, x_cg1.grad, x_eager1.grad, msg=f"Step {step} Obj1 x.grad")
            assert_tensors_close(self, x_cg2.grad, x_eager2.grad, msg=f"Step {step} Obj2 x.grad")

            assert_tensors_close(
                self,
                cg_obj1.linear1.weight.grad,
                eager_obj1.linear1.weight.grad,
                msg=f"Step {step} Obj1 linear1.weight.grad",
            )
            assert_tensors_close(
                self,
                cg_obj2.linear1.weight.grad,
                eager_obj2.linear1.weight.grad,
                msg=f"Step {step} Obj2 linear1.weight.grad",
            )

            opt_eager1.step()
            opt_eager1.clear_grad()
            opt_cg1.step()
            opt_cg1.clear_grad()
            opt_eager2.step()
            opt_eager2.clear_grad()
            opt_cg2.step()
            opt_cg2.clear_grad()


class ConfigLayerEager(nn.Layer):
    def __init__(self, dim=16):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, apply_relu=True):
        return F.relu(self.proj(x)) if apply_relu else self.proj(x)


class ConfigLayerCG(nn.Layer):
    def __init__(self, dim=16):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    @autocudagraph(
        warmup_steps=1,
        max_graphs=2,
        dispatch_key_fn=lambda kw: (kw["x"].shape[0], kw["apply_relu"]),
    )
    def forward_dynamic(self, x, apply_relu=True):
        return F.relu(self.proj(x)) if apply_relu else self.proj(x)


class TestConfigurations(BaseTest):
    def tearDown(self):
        ConfigLayerCG.forward_dynamic.clear_cache()

    def test_dispatch_key_and_max_graphs_fallback(self):
        model_cg = ConfigLayerCG()
        model_eager = ConfigLayerEager()
        model_eager.set_state_dict(model_cg.state_dict())

        configs = [
            ([2, 16], True),  # Graph 1
            ([4, 16], False),  # Graph 2
            ([8, 16], True),  # Fallback to Eager
        ]

        for step in range(6):
            x_size, apply_relu = configs[step % 3]
            x_base = paddle.randn(x_size)

            x_cg = x_base.clone().detach()
            x_cg.stop_gradient = False
            x_eager = x_base.clone().detach()
            x_eager.stop_gradient = False

            loss_cg = model_cg.forward_dynamic(x=x_cg, apply_relu=apply_relu).sum()
            loss_cg.backward()

            loss_eager = model_eager(x_eager, apply_relu=apply_relu).sum()
            loss_eager.backward()

            assert_tensors_close(self, loss_cg, loss_eager, msg=f"Fallback Step {step} Loss")
            assert_tensors_close(self, x_cg.grad, x_eager.grad, msg=f"Fallback Step {step} Grad")


def complex_io_eager(data_dict, scale):
    out = data_dict["inputs"][0] * scale + data_dict["inputs"][1]
    return {"res": [out], "status": "ok"}


@autocudagraph(warmup_steps=1)
def complex_io_cg(data_dict, scale):
    out = data_dict["inputs"][0] * scale + data_dict["inputs"][1]
    return {"res": [out], "status": "ok"}


class TestComplexStructures(BaseTest):
    def tearDown(self):
        complex_io_cg.clear_cache()

    def test_nested_io(self):
        x = paddle.randn([2, 2])
        x.stop_gradient = False
        y = paddle.randn([2, 2])
        y.stop_gradient = False

        x_e = x.clone().detach()
        x_e.stop_gradient = False
        y_e = y.clone().detach()
        y_e.stop_gradient = False

        data_cg = {"inputs": [x, y]}
        data_eager = {"inputs": [x_e, y_e]}

        for _ in range(20):
            out_cg = complex_io_cg(data_cg, 3.0)
            loss_cg = out_cg["res"][0].sum()
            loss_cg.backward()

            out_eager = complex_io_eager(data_eager, 3.0)
            loss_eager = out_eager["res"][0].sum()
            loss_eager.backward()

            assert_tensors_close(self, loss_cg, loss_eager)
            assert_tensors_close(self, x.grad, x_e.grad)

            x.clear_gradient()
            y.clear_gradient()
            x_e.clear_gradient()
            y_e.clear_gradient()

    def test_no_grad_context_alignment(self):
        model_cg = ImplicitLayerCG()
        model_eager = ImplicitLayerEager()

        model_cg.set_state_dict(model_eager.state_dict())

        for step in range(5):
            x = paddle.randn([4, 16])
            x_eager = x.clone().detach()
            x_cg = x.clone().detach()

            with paddle.no_grad():
                out_eager = model_eager(x_eager)
                out_cg = model_cg(x_cg)

            assert_tensors_close(
                self,
                out_cg,
                out_eager,
                msg=f"Step {step} no_grad Forward Output",
            )

        self.assertIsNone(x_cg.grad)
        self.assertIsNone(model_cg.linear1.weight.grad)


def ffn_dispatch(args_dict):
    self = args_dict.get("self")
    x = args_dict.get("x")
    if not isinstance(x, paddle.Tensor):
        return None
    return (tuple(x.shape), args_dict.get("apply_activation"), id(self))


class HeavyFFNBlock(nn.Layer):
    def __init__(self, hidden_dim=256, intermediate_size=1024):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias_attr=False)
        self.norm = nn.LayerNorm(hidden_dim)

    @autocudagraph(warmup_steps=2, max_graphs=10, dispatch_key_fn=ffn_dispatch)
    def forward_heavy(self, x, apply_activation=True):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if apply_activation:
            gate = F.silu(gate)
        intermediate = gate * up
        out = self.down_proj(intermediate)
        out = self.norm(out + x)
        return out

    def forward(self, x, apply_activation=True):
        return self.forward_heavy(x, apply_activation=apply_activation)


class FatModelCG(nn.Layer):
    def __init__(self, num_layers=3, hidden_dim=256):
        super().__init__()
        self.layers = nn.LayerList([HeavyFFNBlock(hidden_dim) for _ in range(num_layers)])

    def forward(self, x, apply_activation=True):
        for layer in self.layers:
            x = layer(x, apply_activation)
        return x


class FatModelEager(nn.Layer):
    def __init__(self, num_layers=3, hidden_dim=256):
        super().__init__()
        self.layers = nn.LayerList()
        for _ in range(num_layers):
            block = nn.Layer()
            block.gate_proj = nn.Linear(hidden_dim, 1024, bias_attr=False)
            block.up_proj = nn.Linear(hidden_dim, 1024, bias_attr=False)
            block.down_proj = nn.Linear(1024, hidden_dim, bias_attr=False)
            block.norm = nn.LayerNorm(hidden_dim)
            self.layers.append(block)

    def forward(self, x, apply_activation=True):
        for layer in self.layers:
            gate = layer.gate_proj(x)
            up = layer.up_proj(x)
            if apply_activation:
                gate = F.silu(gate)
            out = layer.down_proj(gate * up)
            x = layer.norm(out + x)
        return x


class TestFatModel(BaseTest):
    def tearDown(self):
        HeavyFFNBlock.forward_heavy.clear_cache()

    def test_heavy_model_alignment(self):
        model_cg = FatModelCG()
        model_eager = FatModelEager()
        model_cg.set_state_dict(model_eager.state_dict())

        opt_cg = paddle.optimizer.Adam(learning_rate=0.01, parameters=model_cg.parameters())
        opt_eager = paddle.optimizer.Adam(learning_rate=0.01, parameters=model_eager.parameters())

        configs = [True, True, False, True, True]

        for step, apply_activation in enumerate(configs):
            x = paddle.randn([4, 64, 256])
            x_cg = x.clone().detach()
            x_cg.stop_gradient = False
            x_eager = x.clone().detach()
            x_eager.stop_gradient = False
            target = paddle.ones_like(x_cg)

            out_cg = model_cg(x_cg, apply_activation=apply_activation)
            out_eager = model_eager(x_eager, apply_activation=apply_activation)

            assert_tensors_close(self, out_cg, out_eager, msg=f"Step {step} Forward Output")

            loss_cg = F.mse_loss(out_cg, target)
            loss_eager = F.mse_loss(out_eager, target)
            assert_tensors_close(self, loss_cg, loss_eager, msg=f"Step {step} Loss")

            loss_cg.backward()
            loss_eager.backward()

            assert_tensors_close(self, x_cg.grad, x_eager.grad, msg=f"Step {step} Input x.grad")

            for i in range(len(model_cg.layers)):
                cg_layer = model_cg.layers[i]
                eager_layer = model_eager.layers[i]

                assert_tensors_close(
                    self,
                    cg_layer.gate_proj.weight.grad,
                    eager_layer.gate_proj.weight.grad,
                    msg=f"Step {step} Layer {i} gate_proj.weight.grad",
                )
                assert_tensors_close(
                    self,
                    cg_layer.norm.weight.grad,
                    eager_layer.norm.weight.grad,
                    msg=f"Step {step} Layer {i} norm.weight.grad",
                )

            opt_cg.step()
            opt_cg.clear_grad()
            opt_eager.step()
            opt_eager.clear_grad()


class BasicBlock(nn.Layer):
    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    @autocudagraph(warmup_steps=2)
    def forward(self, x):
        return F.gelu(self.norm(self.proj(x)))


class BasicBlockEager(nn.Layer):
    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return F.gelu(self.norm(self.proj(x)))


class TestAdvancedMechanics(BaseTest):
    def tearDown(self):
        BasicBlock.forward.clear_cache()

    def test_gradient_accumulation(self):
        model_cg = BasicBlock()
        model_eager = BasicBlockEager()
        model_cg.set_state_dict(model_eager.state_dict())

        opt_cg = paddle.optimizer.Adam(learning_rate=0.01, parameters=model_cg.parameters())
        opt_eager = paddle.optimizer.Adam(learning_rate=0.01, parameters=model_eager.parameters())

        accumulation_steps = 5

        for step in range(100):
            x = paddle.randn([8, 64])
            x.stop_gradient = False
            x_cg = x.clone().detach()
            x_cg.stop_gradient = False
            x_eager = x.clone().detach()
            x_eager.stop_gradient = False

            out_cg = model_cg(x_cg)
            out_eager = model_eager(x_eager)

            loss_cg = out_cg.sum()
            loss_eager = out_eager.sum()

            loss_cg.backward()
            loss_eager.backward()

            assert_tensors_close(self, x_cg.grad, x_eager.grad, msg=f"Step {step} x.grad")
            assert_tensors_close(
                self,
                model_cg.proj.weight.grad,
                model_eager.proj.weight.grad,
                msg=f"Step {step} proj.weight.grad",
            )

            if (step + 1) % accumulation_steps == 0:
                opt_cg.step()
                opt_eager.step()
                opt_cg.clear_grad()
                opt_eager.clear_grad()

                self.assertTrue(
                    (model_cg.proj.weight.grad == 0).all().item(),
                    msg="CG Grad not cleared!",
                )

    def test_amp_mixed_precision(self):
        model_cg = BasicBlock()
        model_eager = BasicBlockEager()
        model_cg.set_state_dict(model_eager.state_dict())

        for step in range(5):
            with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
                x = paddle.randn([8, 64], dtype="bfloat16")
                x.stop_gradient = False
                x_cg = x.clone().detach()
                x_cg.stop_gradient = False
                x_eager = x.clone().detach()
                x_eager.stop_gradient = False

                out_cg = model_cg(x_cg)
                out_eager = model_eager(x_eager)

            self.assertEqual(out_cg.dtype, out_eager.dtype)

            loss_cg = out_cg.sum()
            loss_eager = out_eager.sum()
            assert_tensors_close(self, loss_cg, loss_eager, msg=f"Step {step} AMP x.grad")

            loss_cg.backward()
            loss_eager.backward()
            assert_tensors_close(self, x_cg.grad, x_eager.grad, msg=f"Step {step} AMP x.grad")

            x_cg.clear_gradient()
            x_eager.clear_gradient()
            model_cg.clear_gradients()
            model_eager.clear_gradients()


def memory_ops_func_eager(x):
    y = x * 2.0
    return y, y.flatten()


@autocudagraph(warmup_steps=2)
def memory_ops_func_cg(x):
    y = x * 2.0
    return y, y.flatten()


class TestEdgeMemoryOps(BaseTest):
    def tearDown(self):
        memory_ops_func_cg.clear_cache()

    def test_tensor_slices_alignment(self):
        for step in range(10):
            big_tensor_eager = paddle.randn([100, 64])
            big_tensor_eager.stop_gradient = False
            big_tensor_cg = big_tensor_eager.clone().detach()
            big_tensor_cg.stop_gradient = False

            slice_eager = big_tensor_eager[10:90:4, 2:20:3]
            slice_cg = big_tensor_cg[10:90:4, 2:20:3]

            out_eager_0, out_eager_1 = memory_ops_func_eager(slice_eager)
            loss_eager = out_eager_0[1::2].sum() + out_eager_1[2::3].sum()
            loss_eager.backward()

            out_cg_0, out_cg_1 = memory_ops_func_cg(slice_cg)
            loss_cg = out_cg_0[1::2].sum() + out_cg_1[2::3].sum()
            loss_cg.backward()

            assert_tensors_close(self, out_cg_0, out_eager_0, msg=f"Step {step} Slice Out 0")
            assert_tensors_close(self, out_cg_1, out_eager_1, msg=f"Step {step} Slice Out 1")
            assert_tensors_close(
                self,
                big_tensor_cg.grad,
                big_tensor_eager.grad,
                msg=f"Step {step} Big Tensor Grad",
            )

            big_tensor_eager.clear_gradient()
            big_tensor_cg.clear_gradient()


@autocudagraph(warmup_steps=1)
def dropout_cg(x):
    y = x * 2.0
    y = F.dropout(y, p=0.5)
    return y + 1.0


class TestGraphLimitationsAndFeatures(BaseTest):
    def tearDown(self):
        dropout_cg.clear_cache()

    def test_dropout_randomness_preservation(self):
        """
        Verify that CUDAGraph preserves Dropout randomness during the Replay phase.
        """
        x_eager = paddle.ones([20, 20])
        x_eager.stop_gradient = False
        x_cg = paddle.ones([20, 20])
        x_cg.stop_gradient = False

        for _ in range(4):
            _ = dropout_cg(x_cg)
        out_cg_capture = dropout_cg(x_cg)

        out_cg_replay_1 = dropout_cg(x_cg)
        out_cg_replay_2 = dropout_cg(x_cg)

        self.assertFalse(
            np.allclose(out_cg_replay_1.numpy(), out_cg_replay_2.numpy()),
            "CG Replay failed to preserve randomness.",
        )


def fake_calc_eager(x):
    x = x * 4
    x = 3 - x
    return x


@autocudagraph(warmup_steps=1, max_graphs=3, dispatch_key_fn=lambda kw: kw["x"].shape[0])
def fake_calc(x):
    x = x * 4
    x = 3 - x
    return x


class TestDispatchNumber(BaseTest):
    def tearDown(self):
        fake_calc.clear_cache()

    def test_max_graphs_limit(self):
        for bs in range(1, 10):
            for _ in range(5):
                with paddle.no_grad():
                    x = paddle.rand([bs, 128])
                    y = fake_calc(x)

        registry = fake_calc.state_registry
        self.assertEqual(len(registry), 3, "Cache size exceeds max_graphs limit!")

        cached_keys = list(registry.keys())
        self.assertIn(1, cached_keys)
        self.assertIn(2, cached_keys)
        self.assertIn(3, cached_keys)
        self.assertNotIn(4, cached_keys, "Fallback trigger failed, cached unexpected key!")


class TestNoGrad(BaseTest):
    def tearDown(self):
        fake_calc.clear_cache()

    def test_no_grad_alignment_with_eager(self):
        for bs in range(1, 8):
            for step in range(5):
                x = paddle.rand([bs, 128])

                x_cg = x.clone().detach()
                x_eager = x.clone().detach()

                with paddle.no_grad():
                    y_cg = fake_calc(x_cg)
                    y_eager = fake_calc_eager(x_eager)

                assert_tensors_close(
                    self,
                    y_cg,
                    y_eager,
                    msg=f"Mismatch at bs={bs}, step={step} under no_grad",
                )

                self.assertIsNone(x_cg.grad)
                self.assertIsNone(x_eager.grad)


class FatResNeXtEager(nn.Layer):
    def __init__(self):
        super().__init__()
        self.backbone = resnext50_64x4d()
        self.head1 = nn.Linear(1000, 512)
        self.act = nn.GELU()
        self.head2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.act(self.head1(x))
        x = self.head2(x)
        return x


class FatResNeXtCG(FatResNeXtEager):
    @autocudagraph(warmup_steps=2)
    def forward(self, x):
        x = self.backbone(x)
        x = self.act(self.head1(x))
        x = self.head2(x)
        return x


class TestEndToEndPerformance(BaseTest):
    def tearDown(self):
        FatResNeXtCG.forward.clear_cache()

    def test_resnext50_accuracy_and_speed(self):
        model_cg = FatResNeXtCG()
        model_eager = FatResNeXtEager()
        model_cg.set_state_dict(model_eager.state_dict())
        model_cg.train()
        model_eager.train()

        opt_cg = paddle.optimizer.Adam(learning_rate=0.001, parameters=model_cg.parameters())
        opt_eager = paddle.optimizer.Adam(learning_rate=0.001, parameters=model_eager.parameters())

        total_steps = 1000
        batch_size = 4

        dummy_inputs = [paddle.rand([batch_size, 3, 224, 224]) for _ in range(total_steps)]
        dummy_targets = [paddle.randint(0, 10, shape=[batch_size]) for _ in range(total_steps)]

        paddle.device.synchronize()
        start_time_cg = time.perf_counter()

        losses_cg = []
        grads_cg = []
        for step in range(total_steps):
            x = dummy_inputs[step].clone().detach()
            y = dummy_targets[step]

            out = model_cg(x)
            loss = F.cross_entropy(out, y)
            loss.backward()

            losses_cg.append(loss.item())
            grads_cg.append(model_cg.head2.weight.grad.clone().cpu())

            opt_cg.step()
            opt_cg.clear_grad()

        paddle.device.synchronize()
        time_cg = time.perf_counter() - start_time_cg

        paddle.device.synchronize()
        start_time_eager = time.perf_counter()

        losses_eager = []
        grads_eager = []
        for step in range(total_steps):
            x = dummy_inputs[step].clone().detach()
            y = dummy_targets[step]

            out = model_eager(x)
            loss = F.cross_entropy(out, y)
            loss.backward()

            losses_eager.append(loss.item())
            grads_eager.append(model_eager.head2.weight.grad.clone().cpu())

            opt_eager.step()
            opt_eager.clear_grad()

        paddle.device.synchronize()
        time_eager = time.perf_counter() - start_time_eager

        assert_tensors_close(
            self,
            paddle.to_tensor(losses_cg),
            paddle.to_tensor(losses_eager),
            msg=f"Loss mismatch at step {step}",
        )
        for step in range(total_steps):
            assert_tensors_close(
                self,
                grads_cg[step],
                grads_eager[step],
                msg=f"Gradient mismatch at step {step} for head2.weight",
            )

        print(f"\n[Performance Benchmark] ResNeXt50 ({total_steps} steps)")
        print(f"Eager Time:      {time_eager:.4f} s")
        print(f"CUDAGraph Time:  {time_cg:.4f} s")
        speedup = time_eager / time_cg if time_cg > 0 else float("inf")
        print(f"Speedup Ratio:   {speedup:.2f}x")

        self.assertLess(
            time_cg,
            time_eager,
            f"Performance Regression! CUDAGraph ({time_cg:.2f}s) is slower than Eager ({time_eager:.2f}s).",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
