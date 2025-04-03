import os


def place_aws_creds_on_remote_machine(sshalias):
    """
    rsync -rP ~/r/creds/dotaws/ ${sshalias}:/home/ubuntu/.aws"

    on the remote, /home/ubuntu/.aws/config
    should say something like:

    [default]
    region = us-west-2
    output = json

    and /home/ubuntu/.aws/credentials
    should say something like:

    [default]
    aws_access_key_id = AKIAblahblahblah
    aws_secret_access_key = XHablahblahblahblahblahblahblah
    """
    print(f"Installing AWS credentials on {sshalias}...")
    os.system(f"rsync -rP ~/r/creds/dotaws/ {sshalias}:/home/ubuntu/.aws")
    print("Done.")


if __name__ == "__main__":
    install_aws_creds(sshalias="g0")