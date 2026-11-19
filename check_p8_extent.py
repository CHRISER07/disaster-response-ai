import fitz

doc = fitz.open('disaster_response_rag/IEEE_FINAL.pdf')
p8 = doc[7]

blocks = p8.get_text('blocks')
col1 = [b for b in blocks if b[0] < 250]
col2 = [b for b in blocks if b[0] >= 250]
c1_max = max(b[3] for b in col1)
c2_max = max(b[3] for b in col2)

print(f"Total pages: {len(doc)}")
print(f"Column 1 max y: {c1_max:.1f} / 718.9 pt")
print(f"Column 2 max y: {c2_max:.1f} / 718.9 pt")

lines = p8.get_text().splitlines()
print(f"Last line on Page 8: {lines[-1]}")
