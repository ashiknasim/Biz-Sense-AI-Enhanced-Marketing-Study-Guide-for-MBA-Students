# Biz-Sense: AI-Powered Learning Companion for Marketing Management

## 📚 Overview
**Biz-Sense** is an AI-driven study assistant designed to simplify and enhance the learning journey for MBA students, particularly in the area of **Marketing Management**. By leveraging advanced AI technologies, this app provides:

- **Personalized learning content** tailored to the user's knowledge level.
- **AI-powered question paper generation** based on academic syllabi.
- **Comprehensive responses to user queries** for effective concept understanding.

## 💡 Key Features
- **Interactive Learning:**
  - Answer user queries with in-depth, structured explanations.
  - Provide practical applications, strategic frameworks, and actionable takeaways.
- **Question Paper Generator:**
  - Generate custom question papers based on provided topics.
  - Align questions with academic requirements, covering foundational, conceptual, and critical thinking aspects.
- **Content Sourcing:**
  - Use ChromaDB for document storage and retrieval.
  - Employ LangChain embeddings for accurate query matching.

## 🛠️ Technology Stack
- **Frontend:** [Streamlit](https://streamlit.io/) for an intuitive user interface.
- **AI Frameworks:**
  - [LangChain](https://www.langchain.com/) for embeddings and vector database integration.
  - **Groq LLM** for AI model inference.
- **Database:** ChromaDB for persistent storage of vectorized documents.
- **Deployment:** Python-based backend.

## 🚀 How It Works
1. **Initialize Database:**
   - Load academic content from PDFs into ChromaDB.
   - Generate sentence embeddings using the `SentenceTransformer`.
2. **Generate Responses:**
   - User queries are matched with the database for context retrieval.
   - The AI model processes the retrieved context and generates detailed responses.
3. **Question Paper Generation:**
   - Topics are mapped to relevant academic content.
   - The AI model formulates a question paper in a predefined format.
4. **User Interaction:**
   - Users can query specific topics or generate question papers directly from the UI.

## 📂 Directory Structure
├── app.py # Streamlit application ├── main.py # Backend logic for model and database ├── chroma_db/ # Persistent vector database (auto-generated) ├── requirements.txt # Python dependencies └── README.md # Project documentation


