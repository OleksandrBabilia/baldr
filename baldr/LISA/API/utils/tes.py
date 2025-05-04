import os
 
from dotenv import load_dotenv
load_dotenv()
print(1, os.getenv('WASABI_ACCESS_KEY'),os.getenv('WASABI_SECRET_KEY'))