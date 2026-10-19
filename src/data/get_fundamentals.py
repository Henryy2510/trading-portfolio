import argparse
from typing import Any, Dict, List

import pandas as pd
import requests

stock_list = [
    "GEE",
    "PVT",
    "PVP",
    "PHP",
    "MSR",
    "BSR",
    "DRI",
    "DGW",
    "VPI",
    "PLC",
    "LAS",
]


def reports(report, symbol, is_quarter):
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.6",
        "Connection": "keep-alive",
        "Origin": "https://senfin.scienceforeconomics.com",
        "Referer": "https://senfin.scienceforeconomics.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-GPC": "1",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Brave";v="155", "Chromium";v="155", "Not(A:Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Linux"',
    }

    if report == "financial-ratio":
        params = {
            "symbol": f"{symbol}",
            "length": "60",
            "is_quarterly": f"{is_quarter}",
        }

        response = requests.get(
            f"https://api-fin-dev.scienceforeconomics.com/analysis/{report}",
            params=params,
            headers=headers,
        )
    else:
        params = {
            "symbol": f"{symbol}",
            "freq": f"{is_quarter}",
        }

        response = requests.get(
            f"https://api-fin-dev.scienceforeconomics.com/analysis/financial-statement/{report}",
            params=params,
            headers=headers,
        )

    return response.json()


def parse_financial_tree(
    data: Dict[str, Any], ticker: str = "", label_as_column: bool = True
) -> pd.DataFrame:
    """
    Hàm đệ quy trích xuất toàn bộ chỉ số tài chính từ response dạng cây (tree-structure).

    Parameters:
    -----------
    data : dict
        Response JSON chứa 'columnDefs' và 'rows'.
    ticker : str, default ""
        Mã cổ phiếu cần thêm vào làm cột dữ liệu.
    label_as_column : bool, default True
        - True: Trả về DataFrame có các Cột là 'label' (chỉ số tài chính), Index là Quý/Thời gian.
                (Chuẩn cho Quant & Time-series).
        - False: Index là 'label', Cột là Quý/Thời gian (Q2/2026, Q1/2026...).

    Returns:
    --------
    pd.DataFrame
    """
    column_defs: List[str] = data.get("columnDefs", [])
    rows: List[Dict[str, Any]] = data.get("rows", [])

    # 1. Hàm đệ quy duyệt qua các tầng subLevel
    def _extract_leaf_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        extracted = []
        for node in nodes:
            datum = node.get("datum", [])

            # Nếu node có dữ liệu thực tế (datum không rỗng)
            if datum and len(datum) > 0:
                extracted.append(
                    {
                        "label": node.get("label"),
                        "dataPointId": node.get("dataPointId"),
                        "datum": datum,
                    }
                )

            # Đệ quy xuống các tầng con nếu có subLevel
            sub_level = node.get("subLevel")
            if sub_level and isinstance(sub_level, list):
                extracted.extend(_extract_leaf_nodes(sub_level))

        return extracted

    # Thu thập toàn bộ các dòng có dữ liệu
    items = _extract_leaf_nodes(rows)

    if not items:
        return pd.DataFrame()

    # 2. Tạo DataFrame gốc (Rows là các chỉ số, Columns là các Quý)
    data_dict = {}
    for item in items:
        lbl = item["label"]
        datum = item["datum"]
        # Đảm bảo số lượng phần tử khớp với columnDefs
        n = min(len(column_defs), len(datum))
        data_dict[lbl] = pd.Series(datum[:n], index=column_defs[:n])

    df = pd.DataFrame(data_dict)

    # Nếu label_as_column = True: Index là Quý, Cột là Chỉ số tài chính
    # Nếu label_as_column = False: Index là Chỉ số tài chính, Cột là Quý
    if not label_as_column:
        df = df.T

    # Thêm cột ticker nếu được truyền vào
    if ticker:
        df.insert(0, "ticker", ticker)

    return df


def save_data(df: pd.DataFrame, file_name: str, output_format: str):
    """Hàm lưu DataFrame theo định dạng tương ứng."""
    if output_format == "csv":
        df.to_csv(file_name, index=True, encoding="utf-8-sig")
    elif output_format == "parquet":
        df.to_parquet(file_name)
    elif output_format == "json":
        df.to_json(file_name, orient="index", force_ascii=False, indent=4)

    print(f"Đã lưu thành công vào file: {file_name}")


def main():
    parser = argparse.ArgumentParser(description="Download thành file để tạo datalake")
    report_type = parser.add_mutually_exclusive_group(required=True)
    report_type.add_argument(
        "--financial-ratio", action="store_true", help="Get financial ratio"
    )
    report_type.add_argument(
        "--balance-sheet", action="store_true", help="Get balance sheet"
    )
    report_type.add_argument(
        "--income-statement", action="store_true", help="Get income statement"
    )
    report_type.add_argument(
        "--cash-flow", action="store_true", help="Get Lưu chuyển tiền tệ"
    )

    parser.add_argument(
        "--output",
        type=str,
        choices=["csv", "parquet", "json"],
        default="csv",
        help="Xuất file csv",
    )
    args = parser.parse_args()

    # Xác định tên báo cáo dựa vào cờ được truyền vào
    if args.financial_ratio:
        report_name = "financial-ratio"
    elif args.balance_sheet:
        report_name = "balance_sheet"
    elif args.income_statement:
        report_name = "income_statement"
    elif args.cash_flow:
        report_name = "cash_flow"
    else:
        report_name = "report"

    # Tự động tạo tên file: ví dụ "financial_ratio.csv"
    output_filename = f"{report_name}.{args.output}"

    dfs = []
    for ticker in stock_list:
        data = reports(report_name, ticker, "true")
        df = parse_financial_tree(data, ticker=ticker)
        if not df.empty:
            dfs.append(df)

    all_df = pd.concat(dfs) if dfs else pd.DataFrame()

    save_data(all_df, file_name=output_filename, output_format=args.output)


if __name__ == "__main__":
    main()
