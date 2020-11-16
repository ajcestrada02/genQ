from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser

def pdf2text_all(stream):
    parser = PDFParser(stream)
    document = PDFDocument(parser)
    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed

    resmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(resmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(resmgr, device)
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
        for obj in device.get_result():
            if isinstance(obj, (LTTextBox, LTTextLine)):
                yield obj.get_text()


def pdf2text_spec(stream,pageFrom,pageTo):
    parser = PDFParser(stream)
    document = PDFDocument(parser)
    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed

    resmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(resmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(resmgr, device)

    for pageNum,page in enumerate(PDFPage.create_pages(document)):
        print(pageNum)
        if pageNum >= pageFrom-1 and pageNum <= pageTo-1:
            interpreter.process_page(page)
            for obj in device.get_result():
                if isinstance(obj, (LTTextBox, LTTextLine)):
                    yield obj.get_text()