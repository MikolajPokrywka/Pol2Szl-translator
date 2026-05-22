import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "google/translategemma-4b-it"
adapter_path = "qloraPol2Szl-translator"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

model.eval()

SOURCE_LANG = "Polish"
SOURCE_CODE = "pol"

TARGET_LANG = "Silesian"
TARGET_CODE = "szl"

text = "Jak się masz?"

prompt = (
    f"<start_of_turn>user\n"
    f"You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to "
    f"{TARGET_LANG} ({TARGET_CODE}) translator. Your goal is to accurately "
    f"convey the meaning and nuances of the original {SOURCE_LANG} text while "
    f"adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.\n"
    f"{text}"
    f"<end_of_turn>\n"
    f"<start_of_turn>model\n"
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=False,
)

translation = (
    response
    .split("<start_of_turn>model\n")[-1]
    .replace("<end_of_turn>", "")
    .strip()
)

print(translation)