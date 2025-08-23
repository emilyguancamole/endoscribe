import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
load_dotenv()

def connect_to_postgres():
    """Connect to the Postgres database"""
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "endoscribedev"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )

def create_tables_if_not_exist():
    """Create all procedure tables (colonoscopy, polyps, eus, ercp) if they don't exist"""
    conn = connect_to_postgres()
    cursor = conn.cursor()

    # todo attending should prob move to a separate table
    # Colonoscopy table
    colonoscopy_tb = """
    CREATE TABLE IF NOT EXISTS colonoscopy_procedures (
        id VARCHAR(255) PRIMARY KEY,
        attending VARCHAR(255),
        indications TEXT,
        bbps_simple VARCHAR(50),
        bbps_right INTEGER,
        bbps_transverse INTEGER
        bbps_left INTEGER
        bbps_total INTEGER,
        extent VARCHAR(255),
        findings TEXT,
        polyp_count INTEGER,
        impressions TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ); """

    polyps_tb = """
    CREATE TABLE IF NOT EXISTS polyps (
        id SERIAL PRIMARY KEY,
        col_id VARCHAR(255) REFERENCES colonoscopy_procedures(id) ON DELETE CASCADE,
        size_min_mm FLOAT,
        size_max_mm FLOAT,
        location VARCHAR(255),
        resection_performed BOOLEAN,
        resection_method VARCHAR(255),
        nice_class INTEGER,
        jnet_class VARCHAR(50),
        paris_class VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );"""

    # EUS table
    eus_tb = """
    CREATE TABLE IF NOT EXISTS eus_procedures (
        id VARCHAR(255) PRIMARY KEY,
        attending VARCHAR(255),
        indications TEXT,
        eus_findings TEXT,
        egd_findings TEXT,
        impressions TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );"""

    # ERCP table
    ercp_tb = """
    CREATE TABLE IF NOT EXISTS ercp_procedures (
        id VARCHAR(255) PRIMARY KEY,
        attending VARCHAR(255),
        indications TEXT,
        egd_findings TEXT,
        ercp_findings TEXT,
        impressions TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    tables = {
        "colonoscopy_procedures": colonoscopy_tb,
        "polyps": polyps_tb,
        "eus_procedures": eus_tb,
        "ercp_procedures": ercp_tb,
    }

    try:
        for table_name, table_sql in tables:
            cursor.execute(table_sql)
            print(f"Table {table_name} created or already exists")
        conn.commit()
        print("All tables created")
    except Exception as e:
        conn.rollback()
        print(f"Error creating tables: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
        

def upsert_extracted_outputs(df, table):
    """Insert or update extracted procedure data from df into specified table"""
    if df.empty:
        print(f"No data to insert into {table}")
        return
    
    conn = connect_to_postgres()
    cursor = conn.cursor()
    
    cols = list(df.columns)
    values = df.values.tolist() # list of lists (each inner list = 1 row of data)

    # Handle polyps table differently (no upsert, just delete/insert to handle potential differences in polyps between extractions)
    if table == "polyps":
        try:
            # Delete existing polyps for these colonoscopies
            col_ids = df['col_id'].unique().tolist()
            delete_query = "DELETE FROM polyps WHERE col_id = ANY(%s)"
            cursor.execute(delete_query, (col_ids,))
            
            # Insert new polyp data
            insert_query = f"""
                INSERT INTO {table} ({', '.join(cols)})
                VALUES %s;
            """
            execute_values(cursor, insert_query, values)
            conn.commit()
            print(f"Successfully upserted {len(df)} records into {table}")
        except Exception as e:
            conn.rollback()
            print(f"Error upserting {table}: {e}")
            raise
        
    # Upsert for baseline procedure data
    else: 
        try:
            #* EXCLUDED= the new row that was attempted to be inserted but conflicted w existing row. we want to use the value from the new row
            insert_query = f"""
                INSERT INTO {table} ({', '.join(cols)}, updated_at)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                {', '.join([f"{col} = EXCLUDED.{col}" for col in cols if col != 'id'])},
                updated_at = CURRENT_TIMESTAMP;
            """

            # Need timestamp placeholder None because df doesn't have timestamp; sql table expects it
            vals_with_timestamp = [row + [None] for row in values] # appends None as a timestamp to each row list (same as row.append(None))
            execute_values(cursor, insert_query, vals_with_timestamp)
            conn.commit()
            print(f"Successfully upserted {len(df)} records into table {table}")

        except Exception as e:
            conn.rollback()
            print(f"Postgres error upserting {table}: {e}")
            raise

    cursor.close()
    conn.close()