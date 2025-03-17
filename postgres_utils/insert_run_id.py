from get_psycopg2_connection import (
     get_psycopg2_connection
)
import json
import uuid

def insert_run_id(
    run_description_jsonable
) -> uuid.UUID:
    """
    Before tracking progress metrics like l1_loss
    during an experiment run,
    we need to insert a row into the training_runs table
    to describe the arguments by which the run was kicked off.
    """

    # Convert Python object to a JSON string
    json_data = json.dumps(run_description_jsonable)

    run_id_uuid = uuid.uuid4()
    
    run_id_str = str(run_id_uuid)

    conn = get_psycopg2_connection(
        database_server_name="zeus",
        database_name="experiment_tracking",
    )

    try:
        with conn:
            with conn.cursor() as cur:
                # We only need to insert 'message' since
                # insertion_time and message_id have defaults
                insert_sql = """
                    INSERT INTO public.training_runs (run_id, description)
                    VALUES (%s, %s);
                """
                cur.execute(
                    insert_sql,
                    (run_id_str, json_data,)
                )
                # Using context managers automatically commits if no exception occurs
    except Exception as e:
        print(f"Error inserting progress message: {e}")
        # Optionally re-raise the exception or handle it
        raise
    finally:
        conn.close()

    return run_id_uuid
