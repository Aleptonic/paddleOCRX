# paddleOCRX

_This is a library built upon paddleOCR to increase accuracy and maintain the output detection structure in case on unstructured documents_

## Initial Usage:
Check the class `DEFAULT` which is a dataclass that is controlling all the file paths that are used in the code and replace the values as follows : -

        single_page_pdf: str = "single_page_pdf"(make a folder with this name and set its path to the variable)
        input_reports: str = "input_reports"(make a folder with this name and set its path to the variable)
        output_reports: str = "output_reports"(replace with the path where u want final output)
        input: str = "input"(replace with the folder path where all documents as pdfs are stored)
Run the script by `python script.py` or any other method of your choice

## Poppler path error:
If you are getting an error ` Unable to get page count. Is poppler installed and in PATH`

Click [here](https://github.com/Aleptonic/PdfSnipper) to resolve the error run the script
