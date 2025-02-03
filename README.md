# PaddleOCRX

PaddleOCRX is an enhanced library built upon [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to improve OCR accuracy while preserving the structure of detected text, especially in unstructured documents.

## 🚀 Features
- Enhances PaddleOCR’s accuracy for unstructured document processing.
- Maintains the document structure while extracting text.
- Handles multi-page PDFs efficiently.

---

## 📂 Setup & Usage

### **1. Configure File Paths**
The class `DEFAULT` (a `dataclass`) manages all file paths used in the project. Update the following paths before running the script:

```python
class DEFAULT:
    single_page_pdf: str = "single_page_pdf"  # Create this folder and set its absolute path here.
    input_reports: str = "input_reports"  # Create this folder and set its absolute path here.
    output_reports: str = "output_reports"  # Set this to your desired output folder path.
    input: str = "input"  # Set this to the folder containing all input PDFs.
```

Ensure all specified directories exist before execution.

---

### **2. Run the Script**
Once paths are set, run the script using:

```bash
python script.py
```
or any preferred method.

---

## 🛠 Troubleshooting

### **Poppler Path Error**
If you encounter:
```
Error processing Report1_page_1.pdf: Unable to get page count. Is poppler installed and in PATH?
```
Poppler is required for PDF processing. Follow these steps to install it:

#### **🔹 Windows**
1. Download [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases).
2. Extract it to a known location, e.g., `C:\Program Files\poppler-xx\`.
3. Add `C:\Program Files\poppler-xx\bin` to your system `PATH`:
   - Open **System Properties** → **Advanced** → **Environment Variables**.
   - Under **System Variables**, find `Path`, click **Edit**, and add `C:\Program Files\poppler-xx\bin`.

#### **🔹 Linux (Debian/Ubuntu)**
Run:
```bash
sudo apt update
sudo apt install poppler-utils
```

#### **🔹 macOS**
Run:
```bash
brew install poppler
```

#### **Verify Installation**
Run:
```bash
pdfinfo -v
```
If Poppler is installed correctly, it will display version details.

**💡 Still facing issues?** Check out [this guide](https://github.com/Aleptonic/PdfSnipper).

---

## 📜 License
This project is licensed under the MIT License.

---

## 🤝 Contributing
Contributions are welcome! Feel free to submit issues and pull requests.

---

## 📞 Support
For issues, open a GitHub issue or reach out to the maintainers.

