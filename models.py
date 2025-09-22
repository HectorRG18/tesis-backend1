import bcrypt
from database import get_db_connection



def create_users_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password TEXT NOT NULL -- Guardamos el hash en lugar del password en texto plano
        );
    ''')
    conn.commit()
    cursor.close()
    conn.close()

def insert_user(username, email, password):
    """Inserta un usuario en la base de datos con la contraseña encriptada."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Encriptar la contraseña antes de guardarla
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    cursor.execute('''
        INSERT INTO users (username, email, password)
        VALUES (%s, %s, %s);
    ''', (username, email, hashed_password))

    conn.commit()
    cursor.close()
    conn.close()
