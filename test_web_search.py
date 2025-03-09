from langchain_community.tools import DuckDuckGoSearchRun

## To get more additional information (e.g. link, source) use DuckDuckGoSearchResults()
from langchain_community.tools import DuckDuckGoSearchResults

## You can also directly pass a custom DuckDuckGoSearchAPIWrapper to DuckDuckGoSearchResults. Therefore, you have much more control over the search results.
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

## EXAMPLE #1
# search = DuckDuckGoSearchRun()
# output = search.run("What is the capital of Texas?")
# print(output)

## EXAMPLE #2: To get more additional information (e.g. link, source) use DuckDuckGoSearchResults()
## You can also just search for news articles. Use the keyword backend="news"
# search = DuckDuckGoSearchResults(backend="news")
# search = DuckDuckGoSearchResults()
# output = search.run("What is the capital of Texas?")
# print(output)

## EXAMPLE #3: You can also directly pass a custom DuckDuckGoSearchAPIWrapper to DuckDuckGoSearchResults. Therefore, you have much more control over the search results.
wrapper = DuckDuckGoSearchAPIWrapper(region="en-us", time="d", max_results=1)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")
output = search.run("What is the capital of Bangladesh?")
print(output)