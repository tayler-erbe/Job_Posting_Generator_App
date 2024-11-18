
# Function to retrieve similar jobs using BERT embeddings and include similarity scores
def retrieve_similar_jobs_bert(new_job_summary, faiss_index, bert_model, top_k=5):
    # Convert new job summary to BERT embedding
    new_embedding = bert_model.encode([new_job_summary], convert_to_tensor=True)
    
    # Use FAISS to find the top k most similar jobs and their scores
    scores, top_k_indices = faiss_index.search(new_embedding.cpu().numpy(), top_k)
    
    # Retrieve corresponding job descriptions and include scores
    similar_jobs = job_content_df.iloc[top_k_indices[0]].copy()
    similar_jobs = similar_jobs.reset_index(drop=True)  # Reset index for consistency
    similar_jobs['Similarity Score'] = pd.Series(scores[0])  # Ensure the column is added
    return similar_jobs

# Streamlit App
st.title("BERT and FAISS-Powered Job Search and Matching Tool")

# User Input for Job Query
user_query = st.text_input("Enter a job query (e.g., 'I need an accountant with 4 years of experience'): ")

# Allow user to select the number of returned job results (k)
top_k = st.number_input("Enter number of job results to return", min_value=1, max_value=20, value=5)

# Retrieve jobs based on the user query
if user_query:
    # Load a pre-trained BERT model for embedding new queries
    bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Display possible jobs retrieved using BERT and FAISS
    similar_jobs_bert = retrieve_similar_jobs_bert(user_query, faiss_index, bert_model, top_k=top_k)
    
    # Ensure the 'Similarity Score' column exists
    if 'Similarity Score' not in similar_jobs_bert.columns:
        st.error("Similarity Score column not found. Please check the retrieval function.")
    else:
        # Format job titles with similarity scores for display in the dropdown menu
        formatted_results = [
            f"{row['Class Title']} (Similarity Score: {row['Similarity Score']:.2f})"
            for _, row in similar_jobs_bert.iterrows()
        ]
        
        # Display a selection box for the user to pick a job title
        selected_job_title_with_score = st.selectbox("Select a job title", formatted_results)
        
        # Extract the job title without the similarity score for further processing
        selected_job_title = selected_job_title_with_score.split(" (Similarity Score:")[0]
        
        # Display full job posting when a job title is selected
        if selected_job_title:
            selected_job = similar_jobs_bert[similar_jobs_bert['Class Title'] == selected_job_title].iloc[0]
            
            # Display job posting details with similarity score
            st.write(f"### {selected_job['Class Title']} (Similarity Score: {selected_job['Similarity Score']:.2f})")
            st.write(f"**Job Duties:** {selected_job['Job Duties']}")
            st.write(f"**Minimum Qualifications:** {selected_job['MAQ']}")
            st.write(f"**Levels of Work:** {selected_job['Levels Of Work']}")
            st.write(f"**Knowledge, Skills, Abilities (KSA):** {selected_job['KSA']}")
