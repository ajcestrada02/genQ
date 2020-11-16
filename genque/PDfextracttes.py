from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import BytesIO
import os


pdfname =  "C:/Users/Doms/Documents/New RL/An Orthonormal Basis for Topic Segmentation in Tutorial Dialogue .pdf"
# PDFMiner boilerplate
rsrcmgr = PDFResourceManager()
sio = BytesIO()
codec = 'utf-8'
laparams = LAParams()
device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)

# Extract text
fp = open(pdfname, 'rb')
for page in PDFPage.get_pages(fp):
    interpreter.process_page(page)
fp.close()

# Get text from StringIO
text = sio.getvalue()

# Cleanup
device.close()
sio.close()

print(text.decode('utf-8'))