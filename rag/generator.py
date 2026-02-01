import torch


class Generator:
    def __init__(self, model, tokenizer, max_new_tokens=100):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        self.model.to(self.device)


    def build_prompt(self, question: str, contexts: list[str]) -> str:
        context_block = "\n".join(contexts)
        return f"""Context:\n{context_block}
                Question:\n{question}
                \nAnswer:"""


    def generate(self, question: str, contexts: list[str], **kwargs) -> str:
        prompt = self.build_prompt(question, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.85,
            top_p=0.85,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()