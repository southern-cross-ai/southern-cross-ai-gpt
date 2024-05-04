from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer

model_name_or_path = r"southern-cross-gpt2\checkpoint-76000"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained(model_name_or_path, pad_token_id=tokenizer.eos_token_id)
#model = model.to("cuda")
while True:
    txt = input("input：")
    input_ids = tokenizer.encode(txt, return_tensors='pt')
    #input_ids = input_ids.to("cuda")
    beam_output = model.generate(
        input_ids,
        max_length=500,
        num_beams=4,
        no_repeat_ngram_size=1,
        length_penalty=1.34,
        early_stopping=True
    )

    print("Output:\n" + 100 * '-')
    output_ =  tokenizer.decode(beam_output[0], skip_special_tokens=True)
    print(output_.replace(" ",""))
