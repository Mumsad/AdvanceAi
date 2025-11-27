import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def chat():
    chat_history_ids = None
    print("Chatbot: Hi! I'm a chatbot. Type 'quit' to exit.")

    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break

        # Encode user message
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # If first message, no history
        if chat_history_ids is None:
            bot_input_ids = new_user_input_ids
        else:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode only the bot's NEW output
        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        print(f"Chatbot: {response}")

chat()
