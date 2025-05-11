from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

wrapper = DuckDuckGoSearchAPIWrapper(region="us", time="m", max_results=5)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news", output_format="list")
results = search.invoke("Is it a good idea to Investing in Tesla stock?")
print(results)
