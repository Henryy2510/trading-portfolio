#!/usr/bin/env bash
set -e

mkdir -p output

BOOK_ID="${1:-trading-and-exchanges-market-microstructure}"
TOKEN="eyJhbGciOiJSUzI1NiIsImtpZCI6IjkwNGIwZjkxYmU2ZWIwODY3ZGFiYmFhYjU4N2RmYzJlMzNmMDBhMjQiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoiSOG7r3UgTmjDonQgTWluaCBRdeG6o24iLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSlliQ0djSjJ1SzVJbjF0TXphVE4xdThRTTdyaF9TX2o3YjZyMkNDS3h1LWpSdzRNc1Q9czk2LWMiLCJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vcXVhbnR2bi1wbGF0Zm9ybSIsImF1ZCI6InF1YW50dm4tcGxhdGZvcm0iLCJhdXRoX3RpbWUiOjE3OTI1OTUwNjgsInVzZXJfaWQiOiJMWVg1Y09rN2JlZExLWFo0Tko0ck01ZkQ5S28xIiwic3ViIjoiTFlYNWNPazdiZWRMS1haNE5KNHJNNWZEOUtvMSIsImlhdCI6MTc5MjU5NTA2OCwiZXhwIjoxNzkyNTk4NjY4LCJlbWFpbCI6Im5oYXRtaW5oaGlwcG9AZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsiZ29vZ2xlLmNvbSI6WyIxMDIyNTM4MDE2NTkzODkwNjAwMzIiXSwiZW1haWwiOlsibmhhdG1pbmhoaXBwb0BnbWFpbC5jb20iXX0sInNpZ25faW5fcHJvdmlkZXIiOiJnb29nbGUuY29tIn19.BHeehwYOjjH_3l77xPQC0PFJcz8ywmWNYVy4qPrLZzC7oPGO-ty8TcqyEEQAMs-gsGhSArLqBT5z239PT-rZ2UlJTmyDfYFGnHzhne-Axm4t8-kyRwNEz3EAN3jOlfRTx7tApQE11qZiS9iShqCcO88byHYu_hXix2h52jExSECNclcow6_jKoOGxiwvJ_e4YPeDr8pZ4XPWBaD3iBmjwVnc5Xh_gPBSnv9kfE6PLwfvkQURPXSKurkGqLG-2CfknEsX2ed9AbFYFifeiSz0U8ia8Uju7MCLj1B5rLYe5Z4sfgkIP-r_Wzcxquev4okZGZeXk5JwWkiz4Eq_9-J8kg"

echo "[*] Lấy URL tải cho sách: $BOOK_ID..."
RESPONSE=$(curl -s "https://76j3y5287b.execute-api.ap-southeast-1.amazonaws.com/v1/library/book/pdf-url?id=${BOOK_ID}" \
  -H 'accept: application/json, text/plain, */*' \
  -H 'accept-language: en-US,en;q=0.7' \
  -H "authorization: Bearer ${TOKEN}" \
  -H 'origin: https://quantvn.com' \
  -H 'referer: https://quantvn.com/' \
  -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36')

PDF_URL=$(echo "$RESPONSE" | grep -o '"pdf_url": *"[^"]*"' | sed 's/"pdf_url": *"//;s/"//')
TITLE=$(echo "$RESPONSE" | grep -o '"title": *"[^"]*"' | sed 's/"title": *"//;s/"//')

if [ -z "$PDF_URL" ]; then
  echo "[-] Lỗi: Không tìm thấy pdf_url. Response:"
  echo "$RESPONSE"
  exit 1
fi

OUT_FILE="output/${TITLE:-$BOOK_ID}.pdf"
echo "[*] Đang tải: $TITLE về $OUT_FILE..."
curl -L "$PDF_URL" -o "$OUT_FILE" --progress-bar

echo "[+] Đã tải xong: $OUT_FILE"
