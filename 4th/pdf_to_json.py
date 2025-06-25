import fitz  # PyMuPDF
import json
from typing import List
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

#  Schema: One entry per invoice row
class StatementEntry(BaseModel):
    invoice_number: str
    assured: str
    reference: str
    insurance_type: str
    due_date: str
    gross: float
    net: float

# Full schema for account statement
class AccountStatement(BaseModel):
    account_number: str
    currency: str
    statement_date: str
    contact: str
    account_name: str
    entries: List[StatementEntry]

# LLM setup
llm = ChatGroq(
    api_key="",  # 🔐 Replace this with your Groq API key
    model_name="llama3-70b-8192",
    temperature=0
)

#  Prompt for extracting data
parser = PydanticOutputParser(pydantic_object=AccountStatement)
prompt = PromptTemplate(
    template="""
You are extracting structured data from a multi-page insurance account statement.

Return the following fields:
- account_number
- currency
- statement_date
- contact
- account_name
- entries: list of invoice rows (invoice_number, assured, reference, insurance_type, due_date, gross, net)

Use this JSON format:
{format_instructions}

PDF Page Text:
{pdf_text}
""",
    input_variables=["pdf_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# ✅ Chain setup
chain = prompt | llm | parser

# ✅ Process each page and collect all entries
def process_multi_page_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_entries = []
    base_info = None

    for i, page in enumerate(doc):
        print(f"🔄 Processing page {i + 1}...")
        text = page.get_text()
        try:
            result = chain.invoke({"pdf_text": text})
            if not base_info:
                # Grab shared info from first successful page
                base_info = {
                    "account_number": result.account_number,
                    "currency": result.currency,
                    "statement_date": result.statement_date,
                    "contact": result.contact,
                    "account_name": result.account_name
                }
            all_entries.extend(result.entries)
        except Exception as e:
            print(f"⚠️ Error on page {i + 1}: {e}")

    if base_info:
        final_output = {**base_info, "entries": [entry.model_dump() for entry in all_entries]}
        with open("account_statement_output.json", "w") as f:
            json.dump(final_output, f, indent=2)
        print("\n✅ JSON saved to account_statement_output.json")
        return final_output
    else:
        print("❌ Could not extract data from any page.")
        return {}

# ✅ Run the full process
if __name__ == "__main__":
    pdf_path = "A_C Statement.pdf"  # Place your PDF in the same folder
    process_multi_page_pdf(pdf_path)
