import os
import time
import requests
import pandas as pd
import warnings
import psycopg2
import psycopg2.extras
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("db.env")
warnings.filterwarnings("ignore")

host=os.getenv("PGHOST")
port=os.getenv("PGPORT")
database=os.getenv("PGDATABASE")
user=os.getenv("PGUSER")
password=os.getenv("PGPASSWORD")

access_token_url = "https://api.dripcapital.com/v1/access/token"
url = "https://api.dripcapital.com/v1/labs/hsn-code/search"
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"

headers = {
    "authorization": "",
    "user-agent": user_agent,
    "accept": "application/json, text/plain, */*"
}

def get_token():
    token_response = requests.get(
        access_token_url,
        headers={"user-agent": user_agent, "accept": "application/json, text/plain, */*"},
        timeout=20
    )
    token_response.raise_for_status()
    token = token_response.json()["token"]
    return f"Token {token}"

headers["authorization"] = get_token()

def insert_records(df):
    if df.empty:
        return
    conn = psycopg2.connect(
        host=host, port=port, database=database, user=user, password=password
    )
    conn.autocommit = True
    cur = conn.cursor()
    insert_sql = """
    INSERT INTO hsn_data2 ( main_hsn_code, hsn_code, description, gst )
    VALUES %s
    ON CONFLICT (hsn_code) DO NOTHING;
    """
    records = df.to_records(index=False).tolist()
    psycopg2.extras.execute_values(cur, insert_sql, records, template=None)
    cur.close()
    conn.close()

def build_hsn_df(hsn_data):
    grp = hsn_data["result"][str(hsn_data["search"])]
    codes = grp["codes"]
    gst = grp.get("gst", "")

    hsn_df = pd.DataFrame(
        [(k, v.strip()) for d in codes for k, v in d.items()],
        columns=["HS Code", "Description"]
    )
    hsn_df["GST%"] = f"{gst}%"

    hsn_df = pd.concat(
        [pd.DataFrame([{
            "HS Code": str(hsn_data["search"]),
            "Description": grp.get("desc", "").strip(),
            "GST%": f"{gst}%"
        }]), hsn_df],
        ignore_index=True
    )
    return hsn_df

def fetch_hsn_data(code: int) -> pd.DataFrame:
    params = {"search": str(code)}
    last_err = None

    for attempt in range(1, 4):  
        try:
            code_response = requests.get(url, headers=headers, params=params, timeout=30)
            if code_response.status_code == 401:
                headers["authorization"] = get_token()
                code_response = requests.get(url, headers=headers, params=params, timeout=30)

            if code_response.status_code in (429, 500, 502, 503, 504):
                last_err = f"http {code_response.status_code}"
                time.sleep(1)
                continue

            if code_response.status_code != 200:
                return pd.DataFrame([{
                    "main_hsn_code": code,
                    "hsn_code": str(code),
                    "description": f"no data (http {code_response.status_code})",
                    "gst": None
                }])

            try:
                hsn_data = code_response.json()
            except ValueError:
                return pd.DataFrame([{
                    "main_hsn_code": code,
                    "hsn_code": str(code),
                    "description": "no data (non-JSON)",
                    "gst": None
                }])

            search_key = str(hsn_data.get("search", code))
            grp = (hsn_data.get("result") or {}).get(search_key)
            if (not grp) or (not grp.get("codes")):
                return pd.DataFrame([{
                    "main_hsn_code": int(search_key),
                    "hsn_code": str(search_key),
                    "description": "no data",
                    "gst": None
                }])

            hsn_df = build_hsn_df(hsn_data)
            hsn_df.insert(0, "main_hsn_code", str(hsn_data["search"]))

            df_db = (
                hsn_df.rename(columns={"HS Code": "hsn_code", "Description": "description", "GST%": "gst"})
                      .assign(gst=lambda d: d["gst"].astype(str).str.rstrip("%").replace("", None))
            )
            df_db["main_hsn_code"] = df_db["main_hsn_code"].astype(int)

            if df_db["gst"].notna().any():
                df_db["gst"] = df_db["gst"].astype(float).astype(int)
            df_db = df_db[["main_hsn_code", "hsn_code", "description", "gst"]]
            return df_db

        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(1)
            continue

    return pd.DataFrame([{
        "main_hsn_code": code,
        "hsn_code": str(code),
        "description": f"no data (after retries: {last_err})",
        "gst": None
    }])

if __name__ == "__main__":
    start, end = 1, 10000
    max_workers = 15

    overall_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        codes = [f"{n:04d}" for n in range(start, end)]
        for df_part in tqdm(ex.map(fetch_hsn_data, codes), total=len(codes), desc="Fetching HSN"):
            if df_part is not None and not df_part.empty:
                insert_records(df_part)

    overall_end = time.time()
    elapsed = overall_end - overall_start
    print(f"Scraping completed in {elapsed/60:.2f} minutes ({elapsed:.2f} seconds).")

