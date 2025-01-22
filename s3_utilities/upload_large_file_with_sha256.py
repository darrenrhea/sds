# written by chatgpt, watch out
import boto3
import hashlib

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def multipart_upload(bucket_name, file_path, object_name=None):
    s3_client = boto3.client('s3')

    checksum = calculate_sha256(file_path)

    if object_name is None:
        object_name = file_path

    # Initialize multipart upload
    response = s3_client.create_multipart_upload(
        Bucket=bucket_name,
        Key=object_name,
        Metadata={'sha256': checksum}
    )

    upload_id = response['UploadId']
    part_number = 1
    parts = []
    chunk_size = 8 * 1024 * 1024  # 8MB chunks

    with open(file_path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break

            # Upload part
            response = s3_client.upload_part(
                Bucket=bucket_name,
                Key=object_name,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=data
            )

            # Store the part information
            parts.append({
                'PartNumber': part_number,
                'ETag': response['ETag']
            })

            part_number += 1

    # Complete the multipart upload
    s3_client.complete_multipart_upload(
        Bucket=bucket_name,
        Key=object_name,
        UploadId=upload_id,
        MultipartUpload={'Parts': parts}
    )

    print(f"File uploaded successfully with SHA-256: {checksum}")

# Example usage
bucket_name = 'your-bucket-name'
file_path = 'path-to-large-file'
multipart_upload(bucket_name, file_path)