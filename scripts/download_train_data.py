import boto3
import json

# should be a file with AWS credentials
with open('../creds.json', 'r') as f:
    js = json.load(f)

ACCESS_KEY = js['ACCESS_KEY']
SECRET_KEY = js['SECRET_ACCESS_KEY']
BUCKET_NAME = 'emprise-feeding-infra-training-data'
PREFIX = 'v1'

session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)
s3 = session.resource('s3')
bucket = s3.Bucket(BUCKET_NAME)

# download all objects with the matching prefix, printing out a count at each iter
count = 0
for obj in bucket.objects.filter(Prefix=PREFIX):
    if count % 1000 == 0:
        print(float(count/30000.))

    # exclude directories
    if '.png' in obj.key:
        bucket.download_file(obj.key, '../data/training/{}'.format(obj.key))
        count += 1