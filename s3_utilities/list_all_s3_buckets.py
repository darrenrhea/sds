import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def list_all_s3_buckets():
    try:
        # Initialize the S3 client
        s3_client = boto3.client('s3')
        
        # Fetch the list of buckets
        response = s3_client.list_buckets()
        
        # Extract bucket names
        buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
        
        return buckets

    except NoCredentialsError:
        print("AWS credentials not found. Please configure your credentials.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials found. Please check your configuration.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    buckets = list_all_s3_buckets()

    for bucket in buckets:
        print(f"{bucket}")
    