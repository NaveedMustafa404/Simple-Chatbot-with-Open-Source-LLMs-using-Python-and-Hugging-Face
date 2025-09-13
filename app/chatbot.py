from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Chatbot:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.history = {}

    def chat(self, session_id: str, user_input: str) -> str:
        
        history = self.history.get(session_id, [])
        history.append(f"User: {user_input}")
        history_string = " ".join(history[-5:])

        inputs = self.tokenizer(history_string, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            repetition_penalty=1.2,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        history.append(f"Bot: {response}")
        self.history[session_id] = history
        return response
