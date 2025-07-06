# import google.generativeai as genai

# # 1. Configure Gemini API key
# genai.configure(api_key="AIzaSyApfAsLZ5_G41N2gg_W3Uar5tlrPOsRjoM")

# # 2. Load Gemini model
# model = genai.GenerativeModel(model_name="models/gemma-3n-e4b-it")

# # 3. Prepare RAG input — retrieved legal context + user query
# retrieved_chunks = f"""
# {RAG-RESPONSE}
# """

# user_query = f"{USER-PROMPT}"

# # 4. Combine into a RAG-style prompt
# rag_prompt = f"""WRITE A BEST PROMPT FOR

# Context:
# {retrieved_chunks}

# Question:
# {user_query}

# Answer:"""
# response = model.generate_content(rag_prompt)

# # 6. Show answer
# print("📜 Gemini RAG Response:\n", response.text)



# # question= "a person slapped his wife what should she do?"
# # rag_prompt = f"""
# # Input:
# # "{question}"

# # Task:
# # – Convert the above civilian-language sentense into formal legal English used in Pakistan.
# # – Rewrite it as a legally structured question.
# # – Use legal terminology and phrasing found in Pakistani statutes, penal codes, and official legal discourse.
# # – Avoid gendered or personal pronouns (e.g., “he,” “she”); prefer neutral legal terms like “the individual,” “the affected party,” etc.
# # – Return only the converted sentence. Do not include explanations, headers, or formatting—only the rewritten sentence.
# # – Length should be up to 50 words and include terminology likely to match legal vector database embeddings.
# # """
# # # 5. Get response from Gemini
