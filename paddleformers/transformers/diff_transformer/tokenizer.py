from paddleformers.transformers.tokenizer_utils import warp_tokenizer
import transformers

DiffTransformerTokenizer = warp_tokenizer(transformers.LlamaTokenizer)