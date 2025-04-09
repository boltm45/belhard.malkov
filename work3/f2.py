import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd


def create_table(db_file, table_name):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Создаем таблицу (если она не существует)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            A_id INTEGER PRIMARY KEY,
            Weight     REAL,
            Size REAL,
            Sweetness REAL,
            Crunchiness REAL,
            Juiciness REAL,
            Ripeness REAL,
            Acidity REAL,
            Quality text 
        )
    ''')
    conn.commit()
    conn.close()

def import_csv_to_sqlite(csv_file, db_file, table_name):
    # Подключаемся к базе данных (или создаем ее)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Открываем CSV файл и читаем его содержимое
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Чтение заголовков (первой строки)

        for row in reader:
            # Вставляем данные в таблицу
            cursor.execute(f'''
                INSERT INTO {table_name} ({', '.join(headers)}) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)

    # Сохраняем изменения и закрываем соединение
    conn.commit()
    conn.close()

def viev(outputs):
  sns.pairplot(outputs)
  plt.show()

def fetch_data_as_dataframe():
    conn = sqlite3.connect('database.db')  # Подключаемся к базе данных
    
    # Используем pandas для выполнения SQL-запроса и получения DataFrame
    df = pd.read_sql_query("SELECT Weight, size, Sweetness, Crunchiness, Juiciness, Ripeness  FROM apples", conn)
    
    conn.close()  # Закрываем соединение
    
    return df


