from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Using small GPT-2 model (recommended)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

print("Model Loaded Successfully!")

def gen_text(prompt_text, tokenizer, model, n_seqs=1, max_length=50):
    encoded_prompt = tokenizer.encode(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt"
    )

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length + len(encoded_prompt),
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=n_seqs
    )

    if len(output_sequences.shape) > 2:
        output_sequences = output_sequences.squeeze()

    generated_sequences = []

    for idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence)

        total_sequence = prompt_text + text[len(tokenizer.decode(
            encoded_prompt[0],
            clean_up_tokenization_spaces=True
        )):]

        generated_sequences.append(total_sequence)

    return generated_sequences


# Test Example
output = gen_text(
    "Artificial intelligence is transforming the world",
    gpt_tokenizer,
    gpt_model,
    max_length=80
)

print("\nGenerated Output:\n")
for o in output:
    print(o)
    print("--------------------------------------------------")
