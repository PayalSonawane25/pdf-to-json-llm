import fitz  # PyMuPDF
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Step 2: Define output schema using exact labels from PDF
class PremiumClosingData(BaseModel):
    Premium_Closing: str = Field(..., description="e.g., CRN0000860487")
    Texel_Reference: str
    Your_Reference: str
    Obligor: str
    Guarantor: str
    Borrower: str
    Client_Legal_Entity: str
    Coverage_Requested: str
    Facility: str
    Policy_Coverage_Period: str
    Document_Date: str
    Due_Date: str
    Base_Premium: float
    Tax: float
    Commitment_and_Other_Fees: float
    Brokerage: float
    Deposit_Premium: float
    Adjustment_Premium: float
    TOTAL: float

parser = PydanticOutputParser(pydantic_object=PremiumClosingData)

# Step 3: Prompt
prompt = PromptTemplate(
    template="""
Extract the following fields from the insurance closing statement using exactly these keys:

- Premium_Closing
- Texel_Reference
- Your_Reference
- Obligor
- Guarantor
- Borrower
- Client_Legal_Entity
- Coverage_Requested
- Facility
- Policy_Coverage_Period
- Document_Date
- Due_Date
- Base_Premium
- Tax
- Commitment_and_Other_Fees
- Brokerage
- Deposit_Premium
- Adjustment_Premium
- TOTAL

Use this schema for formatting:

{format_instructions}

PDF Text:
{statement_text}
""",
    input_variables=["statement_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Step 4: Groq LLM
llm = ChatGroq(
    api_key="",  # Replace with your Groq API key
    model_name="llama3-70b-8192",
    temperature=0
)

# Step 5: Chain
chain = prompt | llm | parser

# Step 6: Run
if __name__ == "__main__":
    pdf_path = "202300019000 CLOSING MOSAIC- INST 9.pdf"  # PDF file name
    text = extract_text_from_pdf(pdf_path)

    print("🧠 Processing PDF with Groq LLM...\n")
    try:
        result = chain.invoke({"statement_text": text})
        if isinstance(result, PremiumClosingData):
            print("✅ Extracted JSON:")
            print(result.model_dump_json(indent=2))

            with open("closing_data.json", "w") as f:
                f.write(result.model_dump_json(indent=2))
                print("\n💾 Saved as closing_data.json")
        else:
            print("⚠️ LLM returned something unstructured:")
            print(result)
    except Exception as e:
        print("❌ Error:", e)
