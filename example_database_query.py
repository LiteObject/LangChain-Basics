from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain

DB_USER = "user"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "city_db"

# Create database connection
try:
    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit(1)


# Create LangChain model
llm = ChatOllama(model="llama3.2", temperature=0)

# Create SQL query chain
chain = create_sql_query_chain(llm=llm, db=db)

# Define query
question = ("Return the top 5 cities (along with their populations "
            "and countries) with the highest population.")

# Execute query and handle errors
try:
    response = chain.invoke({"question": question})
    print(response)
except Exception as e:
    print(f"Error executing query: {e}")
