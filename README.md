# DataBuddy — Your Data, Explained

DataBuddy is a local data analyst you can talk to. Upload a CSV, or open the bundled sample sales file, and ask questions in plain English. The app profiles the table, checks data quality, draws Plotly charts, and answers from Pandas calculations.

Demo Mode is the default. It does not call Amazon Bedrock and it does not need AWS credentials. If you later enable a Bedrock model, the model may choose a tool and rephrase a draft. The numbers still come from the dataset on this machine. A rewrite that introduces a new number is discarded.

## What you can ask

- Give me a quick overview of this dataset.
- Find the top-performing products.
- Show monthly sales trends.
- Check for missing values and duplicates.
- Find unusual values and potential outliers.
- What should I investigate next?

Those cards on the welcome page submit the question. They are not placeholders.

Outliers use Tukey fences: a value is flagged when it sits below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR. That is a screen, not a verdict that the value is wrong.

The sample file, `data/sample_sales.csv`, has `order_date`, `product`, `category`, `region`, `quantity`, `unit_price`, `revenue`, `profit`, and `returned`. Revenue is quantity × unit price. Profit is revenue × a fixed margin for that product, with a few profit cells left blank on purpose.

## Run it on Windows

Open PowerShell in the `databuddy` folder:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The `Set-ExecutionPolicy` line applies only to that PowerShell window. It is there in case activation is blocked.

## Run it on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints. Load **sample sales** from the sidebar if you do not have a CSV handy.

## Tests

From the `databuddy` folder, with the virtual environment active:

```powershell
python -m pytest -q
```

The suite uses synthetic tables and the bundled sample. It does not call AWS.

## Demo Mode

Demo Mode is on until you turn Amazon Bedrock on in **Settings and AWS**. In Demo Mode the app:

- reads the CSV in the session
- routes common questions to a fixed set of tools
- runs those tools with Pandas
- says so when a question does not match a tool

It will not pretend a language model wrote the figures.

## Optional Amazon Bedrock

1. Copy `.env.example` to `.env`.
2. In the Bedrock console, enable one model in your region and copy its model ID. This project does not assume a model is available.
3. Put that region in `AWS_REGION` and the model ID in `BEDROCK_MODEL_ID`.
4. Provide credentials through the standard AWS credential provider chain (`aws configure`, environment variables, or a role). Do not put access keys in `.env` or in the app.
5. Restart Streamlit, open **Settings and AWS**, and press **Test Bedrock connection**.
6. Turn on **Use Amazon Bedrock** only if you are comfortable sending the question and the column schema. Cell values stay local unless you also opt in to a 3-row preview.

A connection is described as verified only after that test request succeeds in the current session. Access denied, throttling, timeouts, an unknown model, and malformed model output fall back to Demo Mode with a visible warning.

Bedrock permissions should be limited to the model you enabled, typically `bedrock:InvokeModel` and `bedrock:Converse` on that model ARN. Confirm the action names in the current Bedrock documentation before you attach a policy.

## Project layout

```text
app.py                  Streamlit entry point
config.py               Environment settings
agent/                  Router, tool registry, narration, Bedrock orchestration
analytics/              CSV loading, quality, metrics, charts, exports
aws/                    Optional Bedrock client
ui/                     Layout and styles
data/sample_sales.csv   Bundled demo data
tests/                  Pytest suite, no AWS required
infrastructure/         Manual deployment notes and a Lambda handler
```

The Lambda handler is not deployed by this repository. See `infrastructure/README.md`.

## Limits

- CSV only, 10 MB by default (`MAX_UPLOAD_MB`), 200,000 rows, 300 columns.
- Tool calls on Linux are interrupted after 30 seconds. Windows relies on the file-size limit because it has no `SIGALRM`.
- The analyst does not run arbitrary Python, shell commands, or model-written SQL.
- It does not forecast. A question about the future gets a historical trend or a clarification.
- PNG chart download needs the optional `kaleido` package. HTML download works without it.
- Cleaned CSV exports drop exact duplicate rows and leave missing values blank.

## Logs

Logs include the tool name and the error type. They do not include cell values, questions, or credentials.
