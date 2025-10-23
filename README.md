
# Lending Club Approval Model (Streamlit)

A simple Streamlit app that trains a logistic regression approval model on a balanced Lending Club dataset (15k approved + 15k rejected).

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected Columns
- Amount Requested (numeric)
- Application Date (date)
- Loan Title (string)
- Zip Code (string or int)
- State (string)
- DTI (numeric)
- EmpLenY (numeric years)
- Approved (0/1 target)

## Deploy on Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to a public GitHub repo.
2. Go to https://share.streamlit.io and connect the repo.
3. Select `app.py` as the entry point.
4. Click Deploy. Your app URL will look like: `https://<github-username>-<repo-name>-<branch>-app.streamlit.app`.
5. Upload your CSV via the sidebar or commit it with the repo (mind data size limits).
```
