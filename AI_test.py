from transformers import AutoTokenizer #type:ignore
from tokenizers import ByteLevelBPETokenizer #type:ignore
from tokenizers.processors import BertProcessing #type:ignore
from torch.utils.data import Dataset #type:ignore

tokenizer = ByteLevelBPETokenizer("./my_ai-vocab.json","./my_ai-merges.txt"
)

tokenizer._tokenizer.post_procesor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

tokenizer.enable_truncation(max_length=512)
tokenizer.encode("sh")
print(tokenizer.encode("knit midi").tokens)