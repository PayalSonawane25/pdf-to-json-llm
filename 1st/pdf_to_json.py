import fitz  # PyMuPDF
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Step 2: Define structured output schema
class Transaction(BaseModel):
    transaction_number: str
    installment_number: str
    transaction_type: str
    status: str
    effective_date: str
    assured: str
    description: str
    amount: float

class StatementData(BaseModel):
    account_number: str
    currency: str
    statement_date: str
    transactions: List[Transaction]

parser = PydanticOutputParser(pydantic_object=StatementData)

# Step 3: Prompt Template
prompt = PromptTemplate(
    template="""
Extract the following structured information from the given bank statement text:
- account number
- currency
- statement date
- transactions with fields: transaction_number, installment_number, transaction_type, status, effective_date, assured, description, and amount.

Return the result in the JSON format using this schema:

{format_instructions}

Statement text:
{statement_text}
""",
    input_variables=["statement_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Step 4: Setup Groq LLM
llm = ChatGroq(
    api_key="",  # Replace this with your Groq API key
    model_name="llama3-70b-8192",   # More stable for structured output
    temperature=0
)

# Step 5: Chain: Prompt → LLM → Parser
chain = prompt | llm | parser

# Step 6: Run the pipeline
if __name__ == "__main__":
    pdf_path = "Account_BA.pdf"  # Ensure this file is in the same folder
    statement_text = extract_text_from_pdf(pdf_path)

    print("🧠 Processing PDF with Groq LLM...\n")
    try:
        result = chain.invoke({"statement_text": statement_text})

        # Check if the result is correctly parsed
        if isinstance(result, StatementData):
            print("✅ Extracted JSON:")
            print(result.model_dump_json(indent=2))

            # Save to file
            with open("output.json", "w") as f:
                f.write(result.model_dump_json(indent=2))
                print("\n💾 Saved as output.json")
        else:
            print("⚠️ Output was not parsed into the expected structure.")
            print("Raw Output:\n", result)

    except Exception as e:
        print("❌ Error while processing:", e)
