import os
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
load_dotenv()

search = GoogleSerperAPIWrapper()
print(search.run("Is it a good idea to Investing in Tesla stock?"))
