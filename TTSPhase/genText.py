import os

SAMPLE_TEXTS = [
    "Hello! How can I assist you with your account today?",
    "I'm sorry to hear you're having trouble. Can you describe the issue?",
    "Your refund has been processed. Is there anything else I can help you with?",
    "To track your order, please provide your order number.",
    "Would you like to upgrade your plan or learn more about our services?",
    "Thank you for contacting customer support. Have a great day!",
    "Let me connect you to a human agent for further assistance.",
    "Can I help you reset your password?",
    "Our latest offers are available on our website. Would you like a link?",
    "Is there anything else I can do for you today?"
]

OUTPUT_DIR = "sampleTexts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, text in enumerate(SAMPLE_TEXTS, start=1):
    filename = f"sample_{idx:02d}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Created {filepath}")