from dotenv import load_dotenv
import os
load_dotenv()
print("GROQ_API_KEY loaded from .env:", bool(os.environ.get("GROQ_API_KEY")))