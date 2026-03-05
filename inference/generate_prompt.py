# pip install datasets
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset('zai-org/LongBench-v2')
# pick a text and truncate by char length, assume each word is approx 4 char
text = dataset["train"][4]['context'][:15029]
# check token num after tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2")
print(len(tokenizer.encode(text, return_tensors="pt")[0]))
# check text
print(text)
# write to file
with open("long_prompt.txt", "w") as file:
    file.write(text)