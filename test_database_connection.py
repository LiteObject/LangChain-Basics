import psycopg2
try:
    conn = psycopg2.connect(
        dbname="city_db",
        user="user",
        password="password",
        host="localhost",
        port="5432"
    )
    print("Connection successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
