import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def chat():
    print("Chatbot: Hi! I'm a chatbot. Type 'quit' to exit.\n")
    chat_history_ids = None

    while True:
        user_input = input("User: ")

        if user_input.lower() == "quit":
            print("Chatbot: Bye! 👋")
            break

        # Tokenize user input with EOS token
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Concatenate with chat history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )

        # Decode only the new tokens
        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        print(f"Chatbot: {response}\n")

chat()
