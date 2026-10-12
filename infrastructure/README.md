# Optional AWS deployment

DataBuddy runs locally with Streamlit. Nothing in this folder creates AWS resources when you install or start the app.

The file `lambda_handler.py` is a manual entry point for the same Pandas tools. It always answers in Demo Mode. It does not call Amazon Bedrock, read Amazon S3, or write logs that contain the CSV. Deploy it only if you want an HTTP API and you accept the cost of the resources below.

The Streamlit app does not call API Gateway. It runs the tools in the same process. A separate frontend should not embed AWS credentials. If you put the API on the public internet, protect it yourself (IAM auth, a private API, or another control you already operate). A wide-open endpoint that accepts CSV text will process whatever is posted to it.

## What you would create, by hand

1. An IAM role for Lambda with a basic execution policy so it can write to CloudWatch Logs.
2. A Lambda function whose handler is `infrastructure.lambda_handler.lambda_handler`, with the project root on `PYTHONPATH`.
3. An API Gateway HTTP API or REST API with `POST /analyze` and Lambda proxy integration.
4. Optional: an S3 bucket only if you later add your own upload storage. The handler in this repo does not read a bucket.
5. Optional: Bedrock model access for the Streamlit app, not for this handler. The handler ignores Bedrock on purpose so a public function cannot silently send schemas to a model.

## Package shape

Lambda needs `pandas`, `numpy`, `plotly`, and, only if you later extend the handler, `boto3`. Streamlit is not imported by the handler. A zip of those libraries can exceed Lambda's unzipped size limit. A container image is the practical package. Do not zip your `.venv` from a development machine and assume it will run.

Example event:

```json
{
  "question": "Give me a quick overview of this dataset.",
  "csv_text": "product,revenue\nA,10\nB,4\n"
}
```

API Gateway proxy events are accepted when the JSON body contains `question` and `csv_text`. Bodies are limited to 1 MB and questions to 2,000 characters. Base64 bodies are rejected.

## IAM

For the Streamlit host, if you enable Bedrock, scope the identity to the one model you enabled. Check the current action names in the Bedrock docs. A typical statement is:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeOneModel",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:Converse"],
      "Resource": "arn:aws:bedrock:REGION::foundation-model/YOUR_MODEL_ID"
    }
  ]
}
```

Do not attach `bedrock:*`, `s3:*`, or `iam:*` for this demo. Do not grant the Lambda function Bedrock access unless you have changed the handler and reviewed what it sends.

If you create a bucket later:

- block public access
- do not use a bucket policy that allows `Principal: "*"`
- add a lifecycle rule that expires objects
- let the function write only the prefix you choose, not the whole account

## CloudWatch

Lambda writes its own log group. In this handler the application log line on failure is only `lambda analysis failed`. It does not log `csv_text` or the question.

After the first invoke:

- open the log group `/aws/lambda/your-function-name`
- set a retention period (14 or 30 days is enough for a demo)
- add an alarm on the Lambda `Errors` metric if you plan to leave the function deployed

## Cleanup

Delete anything you created, in this order, so you do not keep paying for idle resources:

1. Delete the API Gateway API.
2. Delete the Lambda function.
3. Delete the CloudWatch log group for that function.
4. Delete the IAM role after the function no longer uses it.
5. If you created a bucket, empty it and delete it.
6. Bedrock model access does not by itself create a running charge, but invocations do. Turn the model off in the console if you no longer want it available.

Do not run cleanup commands copied from an old account without checking the resource names. This repository does not provision them, so there is nothing for `terraform destroy` or a teardown script to target.
