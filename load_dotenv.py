from dotenv import load_dotenv
import os

load_dotenv()  # loads .env from current directory

print(os.environ["MY_VAR"])
print(os.environ["MY_OTHER_VAR"])

