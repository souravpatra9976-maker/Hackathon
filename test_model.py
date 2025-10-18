from gpt4all import GPT4All


model_path = r"C:\Users\3023149\Desktop\models\TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf"


model = GPT4All(model_name="TinyLlama", model_path=model_path)


response = model.generate("Tell me a joke.")

print("Response:", response)
