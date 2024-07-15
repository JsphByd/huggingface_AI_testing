from transformers import pipeline

fill_mask = pipeline("fill-mask", model="./model", tokenizer="./model")

result = fill_mask("sh<mask>")