import os
import requests

# Cấu hình Token & Headers từ QuantVN
HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-US,en;q=0.7',
    'authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjkwNGIwZjkxYmU2ZWIwODY3ZGFiYmFhYjU4N2RmYzJlMzNmMDBhMjQiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiSOG7r3UgTmjDonQgTWluaCBRdeG6o24iLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSlliQ0djSjJ1SzVJbjF0TXphVE4xdThRTTdyaF9TX2o3YjZyMkNDS3h1LWpSdzRNc1Q9czk2LWMiLCJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vcXVhbnR2bi1wbGF0Zm9ybSIsImF1ZCI6InF1YW50dm4tcGxhdGZvcm0iLCJhdXRoX3RpbWUiOjE3OTI1OTUwNjgsInVzZXJfaWQiOiJMWVg1Y09rN2JlZExLWFo0Tko0ck01ZkQ5S28xIiwic3ViIjoiTFlYNWNPazdiZWRMS1haNE5KNHJNNWZEOUtvMSIsImlhdCI6MTc5MjU5NTA2OCwiZXhwIjoxNzkyNTk4NjY4LCJlbWFpbCI6Im5oYXRtaW5oaGlwcG9AZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZ29vZ2xlLmNvbSI6WyIxMDIyNTM4MDE2NTkzODkwNjAwMzIiXSwiZW1haWwiOlsibmhhdG1pbmhoaXBwb0BnbWFpbC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJnb29nbGUuY29tIn19.BHeehwYOjjH_3l77xPQC0PFJcz8ywmWNYVy4qPrLZzC7oPGO-ty8TcqyEEQAMs-gsGhSArLqBT5z239PT-rZ2UlJTmyDfYFGnHzhne-Axm4t8-kyRwNEz3EAN3jOlfRTx7tApQE11qZiS9iShqCcO88byHYu_hXix2h52jExSECNclcow6_jKoOGxiwvJ_e4YPeDr8pZ4XPWBaD3iBmjwVnc5Xh_gPBSnv9kfE6PLwfvkQURPXSKurkGqLG-2CfknEsX2ed9AbFYFifeiSz0U8ia8Uju7MCLj1B5rLYe5Z4sfgkIP-r_Wzcxquev4okZGZeXk5JwWkiz4Eq_9-J8kg',
    'origin': 'https://quantvn.com',
    'referer': 'https://quantvn.com/',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36',
}

API_ENDPOINT = 'https://76j3y5287b.execute-api.ap-southeast-1.amazonaws.com/v1/library/book/pdf-url'


def download_book(book_id: str, output_dir: str = 'output'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Đang lấy URL tải cho sách: {book_id}...")
    response = requests.get(API_ENDPOINT, params={'id': book_id}, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    title = data.get('title', book_id)
    pdf_url = data.get('pdf_url')

    if not pdf_url:
        raise ValueError(f"Không tìm thấy pdf_url trong response: {data}")

    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
    output_path = os.path.join(output_dir, f"{safe_title}.pdf")

    print(f"[*] Tiêu đề: {title}")
    print(f"[*] Đang tải PDF về: {output_path}...")

    pdf_res = requests.get(pdf_url, stream=True)
    pdf_res.raise_for_status()

    total_size = int(pdf_res.headers.get('content-length', 0))
    downloaded = 0

    with open(output_path, 'wb') as f:
        for chunk in pdf_res.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    print(f"\rTiến độ: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)", end='', flush=True)

    print(f"\n[+] Tải thành công: {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")
    return output_path


if __name__ == '__main__':
    download_book('deep-learning-in-quantitative-trading')
