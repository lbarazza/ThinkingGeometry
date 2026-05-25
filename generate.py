import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-4-E4B-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = model.to("mps")
model.eval()

messages = [{"role": "user", "content": "Solve a quadratic equation of your choice."}]
             #"Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?"}]
prompt = tokenizer.apply_chat_template(messages,
                                       tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=True)

inputs = tokenizer(prompt, return_tensors="pt").to("mps")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=500, do_sample=False)

print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False))