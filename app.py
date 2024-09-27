import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from itertools import chain
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to process the clustering and faculty assignment
def cluster_and_assign_faculty(df_students, df_faculty):
    # TF-IDF vectorization for students' skills and interests
    vectorizer_skills = TfidfVectorizer()
    X_skills = vectorizer_skills.fit_transform(df_students['skills'])
    vectorizer_interests = TfidfVectorizer()
    X_interests = vectorizer_interests.fit_transform(df_students['interest_to_work'])
    
    X_students = hstack([X_skills, X_interests])

    # KMeans clustering
    num_clusters = (len(df_students) + 2) // 3  # Ensures clusters are max 3 students each
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_students['cluster'] = kmeans.fit_predict(X_students)

    clusters = df_students.groupby('cluster')['student_id'].apply(list).tolist()

    # Balancing the clusters with 3 students each
    students = list(chain.from_iterable(clusters))
    balanced_clusters = [students[i:i + 3] for i in range(0, len(students), 3)]

    # Creating a DataFrame for balanced clusters
    balanced_df = pd.DataFrame([(i, student_id) for i, cluster in enumerate(balanced_clusters) for student_id in cluster],
                               columns=['cluster', 'student_id'])
    
    final_df = balanced_df.merge(df_students, on='student_id')
    final_df['Group Number'] = final_df['cluster_x']

    final_data = final_df[['student_id', 'Group Number', 'skills', 'cgpa', 'interest_to_work']]

    # Faculty expertise vectorization and assignment based on cosine similarity
    X_faculty = vectorizer_skills.fit_transform(df_faculty['expertise'])
    cluster_skills = final_df.groupby('Group Number')['skills'].apply(lambda x: ' '.join(x)).reset_index()
    X_cluster_skills = vectorizer_skills.transform(cluster_skills['skills'])
    similarity_matrix = cosine_similarity(X_cluster_skills, X_faculty)

    # Faculty assignment based on highest similarity score
    assigned_faculty = [-1] * num_clusters  
    faculty_assigned = set()  
    for cluster_idx in range(num_clusters):
        similarity_scores = similarity_matrix[cluster_idx]
        faculty_ranking = np.argsort(-similarity_scores)
        for faculty_id in faculty_ranking:
            if faculty_id not in faculty_assigned:
                assigned_faculty[cluster_idx] = df_faculty.iloc[faculty_id]['faculty_id']
                faculty_assigned.add(faculty_id)
                break  
    
    cluster_skills['assigned_faculty_id'] = assigned_faculty
    cluster_skills['assigned_faculty_skills'] = [df_faculty.iloc[idx]['expertise'] for idx in assigned_faculty]

    # Merging the faculty assignments with the final student data
    final_data_with_faculty = final_data.merge(cluster_skills[['Group Number', 'assigned_faculty_id', 'assigned_faculty_skills']], on='Group Number')

    return final_data_with_faculty

# Streamlit App
st.title('Student Clustering and Faculty Assignment')

# Upload input files
students_file = st.file_uploader("Upload the Students CSV File", type="csv")
faculty_file = st.file_uploader("Upload the Faculty CSV File", type="csv")

# Process files if both are uploaded
if students_file and faculty_file:
    try:
        df_students = pd.read_csv(students_file)
        df_faculty = pd.read_csv(faculty_file)

        # Show data for verification
        st.write("### Students Data:")
        st.dataframe(df_students.head())

        st.write("### Faculty Data:")
        st.dataframe(df_faculty.head())

        # Add button to trigger clustering
        if st.button("Run Clustering and Faculty Assignment"):
            final_data_with_faculty = cluster_and_assign_faculty(df_students, df_faculty)

            # Show the resulting dataframe
            st.success("Clustering completed!")
            st.write("### Final Data with Faculty Assignment:")
            st.dataframe(final_data_with_faculty)

            # Provide a download button for the result
            csv = final_data_with_faculty.to_csv(index=False)
            st.download_button(
                label="Download Clustered Data",
                data=csv,
                file_name='clustered_students_with_faculty.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload both the Students and Faculty CSV files to proceed.")
