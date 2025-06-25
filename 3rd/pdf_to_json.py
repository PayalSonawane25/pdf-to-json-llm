import fitz  # PyMuPDF
import json
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Step 2: Define schema using Pydantic
class PDFData(BaseModel):
    Date: str
    Account_Number: str
    Our_Reference: str
    Your_Reference: str
    Reinsured: str
    Original_Insured: str
    Type: str
    Period: str
    Description: str
    Gross_Premium_100_percent: float
    Your_Share_Percent: float
    Your_Share_Amount: float
    Gross_Standard_Commission_Percent: float
    Gross_Standard_Commission_Amount: float
    Colombia_Tax_Percent: float
    Colombia_Tax_Amount: float
    Amount_Payable: float

parser = PydanticOutputParser(pydantic_object=PDFData)

# Step 3: Prompt Template
prompt = PromptTemplate(
    template="""
You are an expert in extracting structured information from insurance PDFs.

Extract and return the following fields as a JSON object:
- Date
- Account_Number
- Our_Reference
- Your_Reference
- Reinsured
- Original_Insured
- Type
- Period
- Description
- Gross_Premium_100_percent
- Your_Share_Percent
- Your_Share_Amount
- Gross_Standard_Commission_Percent
- Gross_Standard_Commission_Amount
- Colombia_Tax_Percent
- Colombia_Tax_Amount
- Amount_Payable

The output must strictly match this schema:

{format_instructions}

PDF Text:
{pdf_text}
""",
    input_variables=["pdf_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Step 4: Set up Groq LLM
llm = ChatGroq(
    api_key="",  # Replace with your actual Groq API key
    model_name="llama3-70b-8192",
    temperature=0
)

# Step 5: Chain together
chain = prompt | llm | parser

# Step 6: Run
if __name__ == "__main__":
    pdf_path = "CLOS - Mosaic Syndicate Services Ltd-09022c5b9726687a.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)

    print("🧠 Processing PDF with schema validation...\n")
    try:
        result = chain.invoke({"pdf_text": extracted_text})
        print("✅ Parsed and Validated JSON:")
        print(result.model_dump_json(indent=2))

        with open("validated_output.json", "w") as f:
            f.write(result.model_dump_json(indent=2))
            print("\n💾 Saved as validated_output.json")

        # 🔄 OPTIONAL: Insert to MongoDB
        # from pymongo import MongoClient
        # client = MongoClient("mongodb://localhost:27017/")
        # db = client["insurance_data"]
        # collection = db["closing_statements"]
        # collection.insert_one(result.model_dump())
        # print("✅ Inserted into MongoDB")

    except Exception as e:
        print("❌ Error during extraction or validation:", e)
