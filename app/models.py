from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import cohere

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

co = cohere.ClientV2()