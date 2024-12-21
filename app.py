# import files
import streamlit as st
import os
import sys
from streamlit import session_state
from io import BytesIO
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from main import query_database, load_model

# Load chromadb with the embeddings
PERSIST_DIRECTORY = "chroma_db"
if os.path.exists(PERSIST_DIRECTORY):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
else:
    st.error("ChromaDB not found. Please ensure the database is initialized.")
    sys.exit(1)

# Loading LLM
llm = load_model()

# Streamlit App Layout
st.set_page_config(page_title="Biz-Sense", layout="wide")
st.title("Biz-Sense")
st.write("An AI-enhanced learning companion")

# streamlit Sidebar-About 
st.sidebar.title("About")
st.sidebar.info(
    """
    **Biz-Sense**  
    A concise, AI-powered study companion designed for MBA students. Biz-Sense symbolizing a keen understanding and intuition in the world of business.  

    **Key Features**:  
    - Academy-oriented syllabus for effective learning.  
    - University question paper generation for exam preparation.    
    - AI-driven content and question generation for practice.

    **Purpose**:  
    Simplify and enhance your learning journey in Marketing Management.

    Powered by advanced AI technology.
    """
    
)
temp="""
KERALA UNIVERSITY OF DIGITAL SCIENCES, INNOVATION AND TECHNOLOGY
SCHOOL OF Digital Humanities and Liberal Arts

FINAL EXAMINATION

YEAR 2024, July

COURSE NAME: MARKETING MANAGEMENT
COURSE CODE: M3440006
COURSE LEVEL: 300

Reg NUMBER: ____________________ TOTAL DURATION: 3 hours
TOTAL MARKS: 100

INSTRUCTIONS TO STUDENTS:
1. Attempt all questions.
2. Section A: 6 x 5 marks = 30 marks
3. Section B: 5 x 10 marks = 50 marks
4. Section C: 2 x 10 marks = 20 marks
5. Draw neat diagrams wherever applicable.

FORMULA, CHEAT SHEETS AND HINTS: NIL

QUESTIONS

Section-A: FOUNDATIONAL KNOWLEDGE ON THE SUBJECT (30%)

Attempt all questions. Each question carries 5 marks.

1. Define the concept of market orientation and its importance in marketing.
2. Explain the role of digital marketing in enhancing customer engagement.
3. Differentiate between a product and a brand with relevant examples.
4. Describe the concept of "customer lifetime value" and its relevance to marketers.
5. Discuss the components of a marketing plan.
6. Explain the importance of the marketing environment in decision-making.

Section-B: CONCEPTUAL UNDERSTANDING ON THE SUBJECT (50%)
Attempt all questions. Each question carries 10 marks.

7. Evaluate the significance of SWOT analysis in marketing strategy development.
8. Discuss the impact of globalization on consumer behavior and marketing practices.
9. Analyze the role of social media influencers in modern marketing campaigns.
10. Compare and contrast push and pull marketing strategies.
11. Illustrate the product lifecycle stages with a real-world example.

Section-C: CRITICAL THINKING ON THE SUBJECT (20%)
Attempt all questions. Each question carries 10 marks.

12. Develop a marketing strategy for launching a new tech product in an emerging market.
13. Analyze how cultural differences influence international marketing strategies.

"""
# Sidebar - Question Paper Generator
st.sidebar.header("Question Paper Generator")
qpaper_topics = st.sidebar.text_area("Enter topics for question paper generation:")
if st.sidebar.button("Generate Question Paper"):
    if qpaper_topics:
        with st.spinner("Generating question paper..."):
            qpaper_context = query_database(db, qpaper_topics)
            if qpaper_context:
                context_text = "\n".join([doc.page_content for doc in qpaper_context])
                prompt = f"Generate a question paper based on the following topics and context:\n{context_text}\n\nTopics: {qpaper_topics} \n\n IN THIS FORMAT {temp}"
                question_paper = llm(prompt)
                st.sidebar.subheader("Generated Question Paper")
                st.sidebar.text_area("", question_paper, height=300)

                # Add download button
                buffer = BytesIO()
                buffer.write(question_paper.encode("utf-8"))
                buffer.seek(0)
                st.sidebar.download_button(
                    label="Download Question Paper",
                    data=buffer,
                    file_name="question_paper.txt",
                    mime="text/plain"
                )
            else:
                st.sidebar.write("No relevant documents found for the provided topics.")
    else:
        st.sidebar.warning("Please enter topics for question paper generation.")

# Main Section - User Query
st.header("What’s on your mind? Let’s learn about it!")
user_query = st.text_input("Enter your query:")
if st.button("Submit Query"):
    if user_query:
        with st.spinner("Fetching relevant documents and generating response..."):
            context = query_database(db, user_query)
            if context:
                context_text = "\n".join([doc.page_content for doc in context])
                prompt = f"Answer the following query based on the context:\n{context_text}\n\nQuery: {user_query}"
                response = llm(prompt)
                st.write(f"### Response:\n{response}")
            else:
                st.warning("No relevant documents found for the query.")
    else:
        st.warning("Please enter a query to generate a response.")

# Footer
st.markdown("---")
st.caption("Powered by Advanced AI technology.")
st.caption("Created By ASHAL NAVEEN EDEN and ASHIK NASIM")