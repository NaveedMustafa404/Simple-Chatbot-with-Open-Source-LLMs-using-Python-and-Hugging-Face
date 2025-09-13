from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

while True:

    user_input = input("> ")
    if user_input.lower() in ["quit", "exit"]:
        break
    
    conversation_history.append(f"User: {user_input}")
    history_string = " ".join(conversation_history[-5:])  
    
    inputs = tokenizer(history_string, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_length=200,
        min_length=5,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        repetition_penalty=1.2,
    )


    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response) 

    conversation_history.append(f"Bot: {response}")
    