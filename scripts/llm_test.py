from model_loader import load_gpt2





if __name__ == '__main__':
    tokenizer, model = load_gpt2()
    prompt = "Hello LLM, I'm a human. How are you doing?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    model_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("LLM output:", model_response)