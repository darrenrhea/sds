import textwrap
from print_red import (
     print_red
)
from get_bucket_name_and_s3_key_from_s3_file_uri import (
     get_bucket_name_and_s3_key_from_s3_file_uri
)
from could_be_an_s3_file_uri import (
     could_be_an_s3_file_uri
)
import boto3
from botocore.exceptions import ClientError

def s3_file_uri_exists(s3_file_uri):
    """
    Return True if the S3 file uri exists, False if it does not.
    Warning: if even the bucket does not exist, this will raise an Exception.
    """
    assert could_be_an_s3_file_uri(s3_file_uri)
    # Parse the S3 URI
    if not s3_file_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI. It must start with 's3://'.")

   
    bucket_name, key = get_bucket_name_and_s3_key_from_s3_file_uri(
        s3_file_uri=s3_file_uri
    )
    
    assert s3_file_uri == f"s3://{bucket_name}/{key}"

    # Initialize S3 client
    s3 = boto3.client('s3')
    
    try:
        # Check if the object exists
        s3.head_object(Bucket=bucket_name, Key=key)
        answer = True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            answer = False
        elif e.response['Error']['Code'] == "403":
            print_red(
                textwrap.dedent(
                    f"""\
                    Access denied?!
                    Could not verify existence of:

                        {s3_file_uri}
                    
                    Oftentimes this means that the bucket itself, i.e.
                    
                        s3://{bucket_name}
                    
                    does not exist or you do not have access to it.
                    """
                )
            )
            raise e
        else:
            raise e
    return answer

