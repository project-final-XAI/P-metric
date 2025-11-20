
import ollama
import time

# Ensure 'image.jpg' is a local image file you want to analyze
try:
    with open('../data/imagenet/n01440764/n01440764_tench.JPEG', 'rb') as f:
        image_data = f.read()
except FileNotFoundError:
    print("Error: image.jpg not found. Please provide a valid image path.")
    exit()

start_time = time.time()

response = ollama.chat(
    model='llama3.2-vision',
    messages=[
        {
            'role': 'user',
            'content': f"What do you see in the picture? Is it a {'shark'} from the imagenet database? answer in yes or no only with one token",
            'images': ['../data/imagenet/n01440764/n01440764_tench.JPEG'] # Provide the path to your image
        }
    ]
)

end_time = time.time()
elapsed_time = end_time - start_time

print("Response message:")
print(response.message.content)
print(f"\nTime taken: {elapsed_time:.2f} seconds")