# DataBuddy — Your Data, Explained

#agents

I built DataBuddy for people who have a CSV and a question, and who should not have to trust a paragraph of made-up statistics. Students, analysts, business owners, and anyone else sitting with an export should be able to ask "which products are underperforming?" and see the ranking, the chart, and the exact formula that produced the number.

## What it does

DataBuddy is a Streamlit workspace. You upload a CSV or open the bundled sample sales file. The app shows the shape of the table, the column kinds, missing values, duplicate rows, numeric summaries, date ranges, and a Pearson correlation matrix when the columns support it. The chat page answers in plain language. Every analytical reply has findings, a table or chart when one helps, a limitation, and a follow-up question you can click.

The sample file has 185 rows and the columns `order_date`, `product`, `category`, `region`, `quantity`, `unit_price`, `revenue`, `profit`, and `returned`. Revenue is quantity times unit price. Profit is revenue times a fixed margin for that product, except for a few profit cells I left blank so the quality report has real gaps to find. There are also four extra duplicate rows and one order with quantity 80.

I checked those claims against the file, not from memory. On this machine the local agent reported:

- Field Backpack has the highest total revenue, 7,308, across 10 products and 185 rows.
- Cedar Candle has the lowest total revenue, 288. That is the answer to "which products are underperforming?"
- Monthly revenue, summed from `order_date`, peaks on the November 2024 period (labeled 2024-11-30) at 2,950 and is lowest on the April 2024 period at 1,148.50. Months with no rows are omitted. They are not filled in as zero.
- The file has 13 missing cells and 4 extra duplicate rows.
- Tukey fences on `quantity` run from 0 to 8 and flag a single value, 80.

Those sentences are only true for this sample. A different file gets a different answer, or a clarification if the column is not there. Asking for "total revenue by customer" on the sample does not invent a customer total. The reply says it cannot find that category and lists `product`, `category`, `region`, and `returned`.

## How the agent is built

The agent is split into three pieces so a language model cannot become the calculator.

The router decides which tool fits the question. In Demo Mode that router is local and explicit. It recognizes overviews, rankings, time trends, missing values, duplicates, outlier screens, correlations, charts, column summaries, and a few follow-ups such as "break that down by region." If the question does not match, or a required column is missing, it says what it can do and stops. It does not forecast, and it does not run arbitrary Python, shell commands, or model-written SQL.

The tool registry is the only thing allowed to touch the dataframe. The ten tools are `dataset_overview`, `get_column_summary`, `calculate_group_metrics`, `calculate_time_trend`, `find_missing_values`, `find_duplicates`, `detect_outliers`, `calculate_correlation`, `create_visualization`, and `suggest_next_questions`. Arguments are checked against a schema: known tool name, known column, and an aggregation or chart type from a fixed list. Extra arguments are rejected. Outliers use Tukey fences, which I state in the reply: values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR. Correlation is Pearson, and the copy says it is not causation.

The narrator writes the reply from the tool result. The "Show how it was calculated" panel lists the columns, the formula, the number of records, and the filter or assumption. "Explain this result" restates that same evidence in everyday language. If the dataset changes, a fingerprint mismatch clears the conversation and hides charts that belonged to the previous file.

Amazon Bedrock is optional. The client uses boto3's Converse API, the region and model ID you set, and the standard credential provider chain. I did not hardcode a model ID, because a model has to be enabled in the account you are using. When Bedrock is on, the model may choose tools and rephrase a draft. The draft is built from the tool output. If the rephrase contains a number that was not in that draft, DataBuddy keeps the verified wording. If the call fails from access denied, throttling, timeout, an unknown model, or malformed output, the same question is answered in Demo Mode and the warning says so. The schema sent upstream is column names, kinds, and missing-value rates. Cell values are included only if you opt in to a 3-row preview. The switch is off by default.

## What I actually ran

I did not deploy this to AWS, and I did not complete a live Bedrock request. No credentials were configured for a successful invoke in this session, so I am not going to claim the cloud path was tested end to end. The client, the error mapping, and the fallback are implemented and covered by tests that feed fake responses, including a model explanation that invents the number 999999. That number is rejected.

What I did run locally:

- `python -m pytest -q` — 41 tests passed. They cover valid and empty CSVs, a malformed file, duplicate columns, missing values, duplicate rows, grouped sums, monthly trends, correlation, missing-column errors, tool argument checks, dataset switching, Demo Mode with Bedrock configuration cleared, and model failure fallback. The suite does not need AWS.
- The sample questions above, through the orchestrator, including a chart on the product ranking, the monthly trend, and the quantity outlier.
- Streamlit's app test harness. Clicking "Give me a quick overview of this dataset." loaded `sample_sales.csv` and rendered the 185-row overview, the Demo Mode banner, and the calculation notes. Clicking "Find the top-performing products." rendered Field Backpack at 7,308. Neither run raised an exception.
- The malformed-file path returns a short message about closed quotation marks and does not include a traceback. The upload page displays that same message.

PNG chart download needs the optional `kaleido` package, which I did not install. HTML chart download is implemented. I would not claim PNG export works on a machine that does not have kaleido.

## What is implemented, and what is only documented

Implemented and used when you run `streamlit run app.py`: the interface, Pandas tools, Plotly charts, session chat, exports for the cleaned CSV, the quality report, and the Markdown summary, plus the Bedrock client if you configure and turn it on.

Documented, not created from this project: Lambda, API Gateway, S3, and CloudWatch alarms. `infrastructure/lambda_handler.py` is a manual entry point for the same local tools. It stays in Demo Mode so a public function cannot silently call a model. `infrastructure/README.md` explains least-privilege IAM, log retention, and cleanup. Running the app does not create those resources.

## Proof of work

Demo URL: [add the URL where you publish DataBuddy]

Screenshot or video: [add a capture of the sample sales chat showing Field Backpack at 7,308, or the quantity outlier of 80]

The local checks above are the evidence I can stand behind. Add your own URL and capture when you publish. Do not reuse these sample figures as if they were measured on a different dataset.

## Profile About

I build DataBuddy, a local-first analyst for people who have a CSV and a question. It profiles the file, checks quality, and answers from a fixed set of Pandas tools with the formula shown beside the result. Amazon Bedrock is optional, and it is not allowed to invent a number. The point is to help students, analysts, and owners see what the rows actually say.