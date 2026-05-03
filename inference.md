**1. Ý định: `card_arrival`** (Hỏi về thời gian nhận thẻ)
*   **Bạn nhập:** `I ordered my new physical card a week ago, when will it arrive at my house?`
*   **Kết quả mong đợi:** `Intent : card_arrival`

**2. Ý định: `change_pin`** (Đổi mã PIN)
*   **Bạn nhập:** `I forgot my PIN code, how can I change it to a new one inside the app?`
*   **Kết quả mong đợi:** `Intent : change_pin`

**3. Ý định: `transaction_charged_twice`** (Bị trừ tiền hai lần cho một giao dịch)
*   **Bạn nhập:** `I bought a coffee this morning but my bank statement shows I was charged two times for the same amount.`
*   **Kết quả mong đợi:** `Intent : transaction_charged_twice`

**4. Ý định: `apple_pay_or_google_pay`** (Hỗ trợ ví điện tử)
*   **Bạn nhập:** `Can I link my bank account to Google Pay on my Android phone?`
*   **Kết quả mong đợi:** `Intent : apple_pay_or_google_pay`

**5. Ý định: `Refund_not_showing_up`** (Tiền hoàn chưa về tài khoản)
*   **Bạn nhập:** `The merchant issued a refund 3 days ago but I still don't see the money in my balance.`
*   **Kết quả mong đợi:** `Intent : Refund_not_showing_up`

**6. Ý định: `terminate_account`** (Đóng tài khoản)
*   **Bạn nhập:** `I want to close my bank account permanently and move my money to another bank.`
*   **Kết quả mong đợi:** `Intent : terminate_account`

**7. Ý định: `wrong_amount_of_cash_received`** (Cây ATM nhả sai tiền)
*   **Bạn nhập:** `The ATM gave me 20 dollars less than what I requested, but my balance was deducted for the full amount.`
*   **Kết quả mong đợi:** `Intent : wrong_amount_of_cash_received`

**8. Ý định: `age_limit`** (Giới hạn độ tuổi mở tài khoản)
*   **Bạn nhập:** `How old do I need to be to open a standard account for my son?`
*   **Kết quả mong đợi:** `Intent : age_limit`
