import pandas as pd
import requests

headers = {
    "sec-ch-ua-platform": '"Linux"',
    "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhdXRoZW5fc2VydmljZSIsImV4cCI6MTc4ODYyMTgxMSwianRpIjoiIiwiaWF0IjoxNzg4NTc4NjExLCJzdWIiOiIxMDAwMDg0NTQyMCIsImN1c3RvZHlJRCI6IjEwNUM3MVFNNzUiLCJ0Y2JzSWQiOiIxMDAwMDg0NTQyMCIsImVtYWlsIjoibmhhdG1pbmgwMTA5MDVAZ21haWwuY29tIiwicm9sZXMiOlsiY3VzdG9tZXIiXSwic2NvcGVzIjpbImFsbDphbGwiLCJzb2NrZXQ6YWxsIl0sInN0ZXB1cF9leHAiOjAsInNvdHBfc2lnbiI6IiIsImNsaWVudF9rZXkiOiIxMDAwMDg0NTQyMC5ZQVZydndJRUxlYVBKdVpSUHpmdCIsInNlc3Npb25JRCI6IjkxNzZiMGYwLTEyMDQtNGVlYS1hZTk3LWIzNjQzYWNiNWM0NiIsImFjY291bnRfc3RhdHVzIjoiMSIsIm90cCI6IiIsIm90cFR5cGUiOiIiLCJvdHBTb3VyY2UiOiJUQ0lOVkVTVCIsIm90cFNlc3Npb25JZCI6IiIsImFjY291bnRUeXBlIjoiUFJJTUFSWSIsInByaW1hcnlTdWIiOiIiLCJwcmltYXJ5Q3VzdG9keUlEIjoiIiwiZW5vdHBfc2lnbiI6IiIsInNxYV9zaWduIjoiIiwiZW5fb3RwIjoiIiwiZW5PVFBUeXBlIjoiIiwiY2FTdGF0dXMiOiJJR05PUkUiLCJjdXNUeXBlIjoiSU5ESVZJRFVBTCIsInRlbmFudCI6InRjYnMiLCJ0Y2JzUm9sZXMiOm51bGx9.lLn_oFQxlB3UpRyh_VRT4KCbjndstZUqpp7Pv4Grg6kVCQPGLCJ8xyyMFmXAoNEUh_qQ82L2nFFPDGVGQUlMPV7Pw9BlD6RqW3xNPJzxd2giLsy2EvNhAel_PXq3GTp9csFcehYsYjlVoQg_ZDyK0f1p1IW7hSf4TbtsP_yitOkQG5RcVrCBGER7-IZ-_8QYr7UJHza0ftq9yXxZiP93F9qSyDRhnsSywTuWub72_6zgRrOOXNT86avD434oBnUU2jrpuaQ2OQMKHKXaBH1J_qdR9NVwDQ3opOvUzSRITuHGtNjW8Cy_fpj12TzH63L8LgJit_FiiAksrqoBmonHSQ",
    "Referer": "https://tcinvest.tcbs.com.vn/",
    "Accept-language": "vi",
    "sec-ch-ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Brave";v="152"',
    "sec-ch-ua-mobile": "?0",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json",
}


# time format là ISO
def history_price(
    ticker,
    time,
):
    params = {
        "ticker": f"{ticker}",
        "type": "stock",
        "resolution": "D",
        "to": f"{time}",
        "countBack": "259",
    }

    response = requests.get(
        "https://apiextaws.tcbs.com.vn/stock-insight/v2/stock/bars-long-term",
        params=params,
        headers=headers,
    )

    data = response.json()
    df = pd.DataFrame(data)
    return df
