from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch

kb_texts = [
    "How to track your Amazon order: Go to Your Orders, select the order, and click Track Package.",
    "How to cancel an order: If your order hasn't shipped, go to Your Orders, select Cancel Items.",
    "How to modify an order: You cannot modify orders after placing them, but you can cancel and reorder if it hasn't shipped.",
    "View order history: Access Your Orders to see all past purchases, delivery dates, and order details.",
    "Delivery estimates: Check estimated delivery dates on the product page or in Your Orders section.",
    "Refund policy: You can return most items within 30 days of delivery for a full refund.",
    "How to initiate a return: Go to Your Orders, select Return or Replace Items, choose reason, and print return label.",
    "Refund processing time: Refunds are typically processed within 2-3 weeks after we receive your return.",
    "Return shipping costs: Most returns are free, but some items may have return shipping fees.",
    "Non-returnable items: Certain items like digital content, gift cards, and personalized items cannot be returned.",
    "Prime membership benefits: Free 2-day delivery, Prime Video, Prime Music, exclusive deals.",
    "Prime membership cost: Monthly or annual subscription options available with various pricing tiers.",
    "How to cancel Prime: Go to Account & Lists, select Prime Membership, and choose End Membership.",
    "Prime free trial: New members can try Prime free for 30 days with full access to all benefits.",
    "Prime Student discount: Students get 6 months free trial and 50% off membership after that.",
    "Report delivery issues: If your package is missing, contact customer service with your order ID.",
    "Package marked delivered but not received: Wait 24-48 hours, check with neighbors, then report to customer service.",
    "Damaged items: Take photos of damage and contact customer service for replacement or refund within 14 days.",
    "Wrong item delivered: Contact customer service immediately for a replacement and return label.",
    "Delivery to Amazon Locker: Select a locker location at checkout and retrieve within 3 days using pickup code.",
    "Payment methods accepted: Credit cards, debit cards, Amazon gift cards, bank accounts, and Amazon Pay.",
    "How to update payment information: Go to Your Account, select Payment Options, and edit or add payment methods.",
    "Promotional credits: Check Your Account for available credits which are automatically applied at checkout.",
    "Subscribe & Save: Get up to 15% off on recurring deliveries of eligible products.",
    "Gift card balance: View balance under Your Account > Gift Cards, or during checkout.",
    "Reset password: Click Forgot Password on sign-in page and follow email instructions.",
    "Update shipping address: Go to Your Addresses to add, edit, or remove delivery addresses.",
    "Enable two-step verification: Increase account security in Login & Security settings.",
    "Manage email preferences: Control promotional emails in Communication Preferences.",
    "Close Amazon account: Contact customer service to permanently close your account.",
    "Read product reviews: Check customer ratings and reviews on product pages for authentic feedback.",
    "Ask product questions: Use the Customer Questions & Answers section on product pages.",
    "Product availability: Out of stock items show expected restock dates when available.",
    "Price matching: Amazon doesn't match competitor prices but offers competitive pricing daily.",
    "Product warranties: Check individual product pages for manufacturer warranty information.",
    "Contact customer service: Use Help & Customer Service to chat, email, or request a phone call.",
    "Customer service hours: Available 24/7 for most issues via chat and phone support.",
    "Report a problem: Use Help section to report issues with orders, products, or account security.",
    "A-to-z Guarantee: Protection for marketplace purchases if items don't arrive or don't match description.",
    "Leave seller feedback: Rate third-party sellers to help other cv. Customers make informed decisions.",
]

docs = [Document(page_content=text) for text in kb_texts]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  
    chunk_overlap=50  
)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,  
        "fetch_k": 10  
    }
)

model_path = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,  
    min_new_tokens=20,   
    do_sample=False,
    num_beams=2,        
    early_stopping=True
)

hf_llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = """Based on the context below, provide a direct answer to the question. Use only the information from the context. In case you dont have the context, tell them that you will escalate to a higher level customer care staff. make sure to greet them everytime they ask something.
Context: {context}

Question: {question}

Direct answer (2-3 sentences):"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=hf_llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=False
)

# Helper function to clean response
def clean_response(response_text):
    """Extract only the answer portion and clean it"""
    text = response_text.strip()
    
    if "Context:" in text:
        text = text.split("Context:")[0].strip()
    if "Question:" in text:
        text = text.split("Question:")[0].strip()
    if "Answer:" in text:
        parts = text.split("Answer:")
        text = parts[-1].strip() if len(parts) > 1 else text
    
    sentences = []
    for sent in text.replace('!', '.').replace('?', '.').split('.'):
        sent = sent.strip()
        if sent and len(sent) > 5:  # Ignore very short fragments
            sentences.append(sent)
            if len(sentences) >= 2:  # Stop after 2 good sentences
                break
    
    if sentences:
        result = '. '.join(sentences)
        if not result.endswith('.'):
            result += '.'
        return result
    
    return text if text else "I don't have enough information to answer that question. I will escalate this issue to a senior staff. Thank you for your time."


def ask_query(query_text):
    """Ask query and show answer with sources"""
    response = qa.invoke(query_text)
    cleaned_answer = clean_response(response['result'])
    
    print("Query:", query_text)
    print("Answer:", cleaned_answer)
    
    if response.get('source_documents'):
        print("Sources used:")
        for i, doc in enumerate(response['source_documents'][:2], 1):
            print(f"  {i}. {doc.page_content[:80]}...")
    print()
    
    return cleaned_answer


