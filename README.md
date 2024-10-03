
# Job Search App - Readme

## Overview
This repository contains two different job search applications, each built using a distinct methodology for job retrieval and matching. Both applications are hosted and deployed using **Streamlit**, a free and open-source framework for building web applications quickly in Python. The apps allow users to query job descriptions based on specific terms and find relevant job titles using two different models: **BERT with FAISS** and **TF-IDF with Cosine Similarity**. Below is a breakdown of the repository's contents and instructions on how to run the applications.

---

## Files and Folders

### 1. **requirements.txt**
   This file lists all the Python packages and dependencies required to run the applications. You can install them using the following command:

   ```
   pip install -r requirements.txt
   ```

   It ensures that all necessary libraries (such as `streamlit`, `sklearn`, `transformers`, and `faiss`) are available in your environment.

### 2. **job_content.csv**
   - A simplified dataset containing relevant information about various job titles and descriptions. 
   - This data was extracted from a larger dataset called **Class Index Export 03.28.2024.xlsx**, which originated from SUCCS (State Universities Civil Service System). 
   - It includes only the columns needed for the job search functionality.

### 3. **Class Index Export 03.28.2024.xlsx**
   - The original dataset from SUCCS.
   - This file has multiple sheets and contains extensive information about job classifications, roles, and responsibilities. 
   - While not directly used in the app, this file is provided for reference or additional context.

### 4. **TF-IDF_Based_Job_Search_with_TxtDoc_Download_App.py**
   - This is the Python script that powers the **TF-IDF Based Job Search App**.
   - It utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** to vectorize job descriptions and **Cosine Similarity** to match user queries with relevant job titles.
   - Suitable for faster, keyword-driven searches where the focus is on term matching rather than deeper contextual understanding.
   
### 5. **BERT_and_FAISS_Powered_Job_Search_and_Matching_App.py**
   - This is the Python script for the **BERT and FAISS Powered Job Search App**.
   - It uses **BERT (Bidirectional Encoder Representations from Transformers)** to generate embeddings of job descriptions and **FAISS (Facebook AI Similarity Search)** to perform fast similarity searches between user queries and jobs.
   - This model is better suited for complex, context-aware searches where understanding the relationships between terms is critical.

### 6. **bert_embeddings.npy**
   - This file contains precomputed **BERT embeddings** for the job descriptions in the dataset.
   - BERT embeddings capture the context and semantic meaning of the text, allowing for a deeper understanding of the relationships between terms in the job descriptions and user queries.
   - In the app, these embeddings are loaded and used with FAISS to quickly find job descriptions that are most similar to the user's query.

### 7. **faiss_index.index**
   - This file contains the pre-built **FAISS index**, which is used for fast similarity searches on the BERT embeddings.
   - FAISS (Facebook AI Similarity Search) is a library that enables efficient similarity searches over large datasets of embeddings.
   - The index is loaded into the app, allowing it to search for the top-k most relevant jobs based on the BERT embeddings of the user's query.

---

## Applications

### 1. **BERT + FAISS Powered Job Search App**
   - **Model:** BERT for embeddings, FAISS for similarity search.
   - **How it works:** 
     - BERT is used to generate embeddings (vector representations) for both the job descriptions and the user's query. 
     - FAISS then performs a fast similarity search to match the most relevant jobs based on these embeddings.
   - **Advantages:** 
     - BERT captures the semantic meaning behind words, making this method better at understanding the relationships between terms (e.g., recognizing that "finance manager" and "accounting supervisor" are related roles).
   - **Use case:** Ideal for complex and context-aware searches where understanding the meaning behind the query is important.
   - **How to run:**
     ```
     streamlit run BERT_and_FAISS_Powered_Job_Search_and_Matching_App.py
     ```

### 2. **TF-IDF Based Job Search App**
   - **Model:** TF-IDF for vectorization, Cosine Similarity for matching.
   - **How it works:** 
     - TF-IDF assigns weights to terms based on their frequency in the dataset, emphasizing unique terms over common ones.
     - Cosine similarity is then used to compare the user's query with the job descriptions to find the most relevant matches.
   - **Advantages:** 
     - This approach is faster and simpler than the BERT-based model, but it is less context-aware. It focuses on individual word importance rather than semantic meaning.
   - **Use case:** Suitable for straightforward keyword-based searches where the importance of specific terms is more relevant than the overall context.
   - **How to run:**
     ```
     streamlit run TF-IDF_Based_Job_Search_with_TxtDoc_Download_App.py
     ```

---

## Summary of Differences

- **BERT + FAISS Powered Job Search App:**
   - More sophisticated, context-aware model that captures the semantic meaning behind job descriptions and queries.
   - Ideal for more complex job searches where understanding the context is crucial.
   
- **TF-IDF Based Job Search App:**
   - Simpler, keyword-driven approach that focuses on term frequency rather than context.
   - Faster but less nuanced than the BERT model, making it ideal for straightforward, keyword-based searches.

---

## How to Get Started

1. **Clone the Repository:**
   ```
   git clone https://github.com/your-repository-link.git
   ```
2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Run the App:**
   - For the TF-IDF based app:
     ```
     streamlit run TF-IDF_Based_Job_Search_with_TxtDoc_Download_App.py
     ```
   - For the BERT + FAISS based app:
     ```
     streamlit run BERT_and_FAISS_Powered_Job_Search_and_Matching_App.py
     ```

Once deployed, the applications will open in a browser window, and you can begin entering queries to search for relevant job descriptions based on the dataset provided.

---

## Additional Notes
- **Streamlit Hosting:** Streamlit is free and open-source, allowing for easy deployment and sharing of apps without the need for complex server configurations.
- **Job Dataset:** The job data used in these apps is based on a simplified version of the SUCCS dataset, specifically curated to showcase job descriptions and classifications for search functionality.

