print('====== Используем pdfplumber ======')

import pdfplumber

with pdfplumber.open("two_tables_example.pdf") as pdf:
    for p in pdf.pages:
        for t in p.extract_tables():
            for r in t:
                print(r)


print('====== Используем camelot-py ======')

import camelot

a = camelot.read_pdf("two_tables_example.pdf")
print(a[0].df)
print(a[1].df)


print('====== Используем PyMuPDF ======')

import fitz 

d = fitz.open("simple_example.pdf")
for p in d:
    t = p.get_text("dict")
    print(t)
