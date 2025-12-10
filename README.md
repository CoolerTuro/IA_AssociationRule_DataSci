# 🧠 Association Rule Mining Project  
**Market Basket Analysis using Apriori Algorithm**

โปรเจกต์นี้เป็น Mini Project ในวิชา Data Science  
เน้นการวิเคราะห์ความสัมพันธ์ของสินค้าในตะกร้าซื้อ (Market Basket Analysis)  
ด้วยเทคนิค **Apriori Algorithm + Association Rules**

---

## 📂 Project Structure

```text
IA_AssociationRule_DataSci/
│
├─ notebooks/                 # Notebook วิเคราะห์ระบบ
├─ src/                       # โค้ดแยกส่วน → data / models / utils
├─ reports/                   # สรุปผลและรูปภาพ
├─ requirements.txt           # ไลบรารีที่ใช้
└─ README.md                  # คำอธิบายโปรเจกต์

🚀 วิธีใช้งาน
1) ใช้บน Google Colab (แนะนำ)
!pip install -r https://raw.githubusercontent.com/CoolerTuro/IA_AssociationRule_DataSci/main/requirements.txt


เปิด Notebook ที่:

notebooks/AssociationRule_Lab.ipynb

📊 ขั้นตอนการวิเคราะห์

โหลดข้อมูล Market Basket

แปลงข้อมูลเป็นรายการสินค้า (transactions)

One-hot Encoding ด้วย TransactionEncoder

สร้าง Frequent Itemsets (Apriori)

สร้าง Association Rules

คัดเลือกกฎตาม Support / Confidence / Lift

สรุปผลเป็นรายงานใน reports/summary.md

📈 ผลลัพธ์ที่ได้

Frequent Itemsets ที่ Support สูง

Association Rules หลายรูปแบบ

เลือก 3 กฎตามรูปแบบที่โจทย์กำหนด (1→1, 2→1, 2→2)

กฎที่มี Lift สูงสามารถนำไปใช้ทางธุรกิจได้จริง เช่นการจัดโปรหรือวางสินค้า

🧑‍💻 ผู้จัดทำ

Prachaya Laosri (65011212183) 
Data Science Mini Project – Association Rule Mining
