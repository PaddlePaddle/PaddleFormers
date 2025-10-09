import sys
import torch
torch.set_default_device('cuda')
import paddle
import numpy as np
paddle.set_printoptions(linewidth=160)
torch.set_printoptions(precision=8, linewidth=160, sci_mode=False)
from transformers import AutoTokenizer, Qwen3NextForCausalLM
from paddleformers.transformers import AutoModelForCausalLM

model_path = "/home/work/Qwen/Qwen3-Next-80B-A3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

input_ids = [tokenizer("今天是周五,明天是周六,后天是周日,那么大后天是周几?")["input_ids"]]
print('input_ids:', input_ids)

############################################################################
if sys.argv[1] == 'pt':
    print('loading torch model')
    pt_model = Qwen3NextForCausalLM.from_pretrained(model_path, dtype="bfloat16")
    pt_model.eval()

    input_ids_pt = torch.tensor(input_ids).long()
    with torch.no_grad():
        pt_out = pt_model(input_ids_pt, return_dict=True)

    logits_pt = pt_out.logits.float().cpu().numpy()
    np.save("/work/torch.logits.npy", logits_pt)

    print('logits_pt:', logits_pt.shape, logits_pt.dtype)
    print(logits_pt)
    exit()

###########################################################################
if sys.argv[1] == 'pd':
    print('loading paddle model')
    pd_model = AutoModelForCausalLM.from_pretrained(model_path, convert_from_hf=True) # , dtype="float32")
    pd_model.eval()

    input_ids_pd = paddle.to_tensor(input_ids)
    with paddle.no_grad():
        pd_out = pd_model(input_ids_pd, return_dict=True)

    logits_pd = pd_out.logits.float().numpy()
    np.save("/work/paddle.logits.npy", logits_pd)

    print('logits_pd:', logits_pd)
    exit()
