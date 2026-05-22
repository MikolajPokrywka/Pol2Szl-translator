# Pol2Szl-translator

Polish → Silesian translator based on `google/translategemma-4b-it`, fine-tuned using QLoRA.

## Files

- `qloraPol2Szl-translator.zip` — QLoRA adapter weights
- `inference.py` — inference script
- `evaluation_set.tsv` — custom evaluation dataset

## Usage

```bash
pip install torch transformers peft accelerate bitsandbytes
unzip qloraPol2Szl-translator.zip
python inference.py
```

## Notes

- Requires the original `google/translategemma-4b-it` model
- Uses QLoRA for efficient fine-tuning
