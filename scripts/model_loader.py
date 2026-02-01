from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = 'gpt2'


def load_gpt2(cache_dir_path=None):
    if cache_dir_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir_path)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=cache_dir_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model




if __name__ == '__main__':
    tokenizer, model = load_gpt2()
    print("GPT-2 small and its tokenizer loaded successfully")