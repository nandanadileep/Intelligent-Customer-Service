from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch
import numpy as np
from collections import deque
import json

# Knowledge base texts
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
    "Leave seller feedback: Rate third-party sellers to help other customers make informed decisions.",
]

docs = [Document(page_content=text) for text in kb_texts]
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})

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

qa = RetrievalQA.from_chain_type(
    llm=hf_llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=False
)


class GRPOOptimizer:
    """Group Relative Policy Optimization for improving responses"""
    
    def __init__(self, group_size=4, learning_rate=0.001, gamma=0.99):
        self.group_size = group_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.response_history = deque(maxlen=100)
        self.reward_baseline = 0.5
        
    def calculate_reward(self, response, query, context):
        """Calculate reward score for a response"""
        reward = 0.0
        
        word_count = len(response.split())
        if 15 <= word_count <= 50:
            reward += 0.3
        elif word_count < 10:
            reward -= 0.2
        
        greetings = ["hello", "hi", "greetings", "thank you", "thanks"]
        if any(greeting in response.lower() for greeting in greetings):
            reward += 0.2
        
        context_keywords = set(context.lower().split())
        response_keywords = set(response.lower().split())
        overlap = len(context_keywords & response_keywords)
        if overlap > 3:
            reward += 0.3
        elif overlap > 1:
            reward += 0.15
        
        if response.strip().endswith(('.', '!', '?')):
            reward += 0.1
        
        sentences = response.split('.')
        unique_ratio = len(set(sentences)) / max(len(sentences), 1)
        reward += 0.1 * unique_ratio
        
        uncertain_phrases = ["don't have", "not sure", "unclear"]
        if any(phrase in response.lower() for phrase in uncertain_phrases):
            if overlap < 2:  
                reward -= 0.1
        
        return max(0.0, min(1.0, reward))
    
    def generate_response_group(self, query, qa_chain, num_responses=None):
        """Generate multiple responses for the same query"""
        if num_responses is None:
            num_responses = self.group_size
        
        responses = []
        for _ in range(num_responses):
            response = qa_chain.invoke(query)
            responses.append(response['result'])
        
        return responses
    
    def compute_group_advantages(self, rewards):
        """Compute advantages using group normalization"""
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8
        
        advantages = (rewards - mean_reward) / std_reward
        
        self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * mean_reward
        
        return advantages, mean_reward
    
    def select_best_response(self, query, context, responses, rewards):
        """Select the best response based on rewards"""
        best_idx = np.argmax(rewards)
        best_response = responses[best_idx]
        best_reward = rewards[best_idx]
        
        self.response_history.append({
            'query': query,
            'context': context,
            'response': best_response,
            'reward': best_reward
        })
        
        return best_response, best_reward
    
    def get_performance_stats(self):
        """Get statistics on response performance"""
        if not self.response_history:
            return None
        
        rewards = [item['reward'] for item in self.response_history]
        return {
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'std_reward': np.std(rewards),
            'num_samples': len(rewards)
        }


grpo = GRPOOptimizer(group_size=4, learning_rate=0.001)


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
        if sent and len(sent) > 5:
            sentences.append(sent)
            if len(sentences) >= 3:
                break
    
    if sentences:
        result = '. '.join(sentences)
        if not result.endswith('.'):
            result += '.'
        return result
    
    return text if text else "I don't have enough information to answer that question. I will escalate this issue to a senior staff. Thank you for your time."


def ask_query_with_grpo(query_text, use_grpo=True):
    """Ask query with GRPO optimization"""
    
    retrieved_docs = retriever.get_relevant_documents(query_text)
    context = " ".join([doc.page_content for doc in retrieved_docs[:3]])
    
    if use_grpo:
        print(f"Generating {grpo.group_size} candidate responses...")
        responses = grpo.generate_response_group(query_text, qa)
        
        cleaned_responses = [clean_response(r) for r in responses]
        
        rewards = [grpo.calculate_reward(r, query_text, context) for r in cleaned_responses]
        
        advantages, mean_reward = grpo.compute_group_advantages(rewards)
        
        best_response, best_reward = grpo.select_best_response(
            query_text, context, cleaned_responses, rewards
        )
        
        print(f"\nQuery: {query_text}")
        print(f"Answer: {best_response}")
        print(f"\nGRPO Stats:")
        print(f"  Best Reward: {best_reward:.3f}")
        print(f"  Mean Reward: {mean_reward:.3f}")
        print(f"  Reward Range: [{min(rewards):.3f}, {max(rewards):.3f}]")
        
        stats = grpo.get_performance_stats()
        if stats and len(grpo.response_history) > 5:
            print(f"\nOverall Performance (last {stats['num_samples']} queries):")
            print(f"  Average Reward: {stats['avg_reward']:.3f}")
            print(f"  Std Deviation: {stats['std_reward']:.3f}")
        
        return best_response
    
    else:
        response = qa.invoke(query_text)
        cleaned_answer = clean_response(response['result'])
        
        print(f"\nQuery: {query_text}")
        print(f"Answer: {cleaned_answer}")
                    
        return cleaned_answer


if __name__ == "__main__":
    print("=== Testing RAG with GRPO Optimization ===\n")
    
    test_queries = [
        "How do I track my order?",
        "What is your refund policy?",
        "Can I cancel my Prime membership?",
        "What should I do if my package is damaged?"
    ]
    
    for query in test_queries:
        print("\n" + "="*70)
        ask_query_with_grpo(query, use_grpo=True)
        print("="*70)
    
    print("\n\nFinal Performance Summary:")
    stats = grpo.get_performance_stats()
    if stats:
        print(json.dumps(stats, indent=2))