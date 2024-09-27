import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from itertools import chain
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Streamlit app title
st.title("Student Clustering and Faculty Assignment")

# File uploader for students and faculty data
st.subheader("Upload Student and Faculty CSV Files")

# Upload student data
student_file = st.file_uploader("Upload Student Data CSV", type=["csv"])
# Upload faculty data
faculty_file = st.file_uploader("Upload Faculty Data CSV", type=["csv"])

if student_file and faculty_file:
    # Load student data
    df_students = pd.read_csv(student_file)
    st.write("Student Data:")
    st.dataframe(df_students.head())

    # Load faculty data
    df_faculty = pd.read_csv(faculty_file)
    st.write("Faculty Data:")
    st.dataframe(df_faculty.head())

    # Process data
    if st.button("Run Clustering and Assignment"):
        try:
            # Vectorize student skills and interests
            vectorizer_skills = TfidfVectorizer()
            X_skills = vectorizer_skills.fit_transform(df_students['skills'])

            vectorizer_interests = TfidfVectorizer()
            X_interests = vectorizer_interests.fit_transform(df_students['interest_to_work'])

            # Combine skills and interests
            X_students = hstack([X_skills, X_interests])

            # Define number of clusters (3 students per cluster)
            num_clusters = (len(df_students) + 2) // 3  
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df_students['cluster'] = kmeans.fit_predict(X_students)

            # Check if clustering was successful
            if df_students['cluster'].isnull().any():
                st.error("Clustering did not produce valid results. Please check your input data.")
            else:
                # Create balanced clusters
                clusters = df_students.groupby('cluster')['student_id'].apply(list).tolist()
                students = list(chain.from_iterable(clusters))
                balanced_clusters = [students[i:i + 3] for i in range(0, len(students), 3)]

                # Create a DataFrame for final data
                balanced_df = pd.DataFrame([(i, student_id) for i, cluster in enumerate(balanced_clusters) for student_id in cluster],
                                           columns=['cluster', 'student_id'])
                final_df = balanced_df.merge(df_students, on='student_id', how='left')

                # Prepare to assign faculty based on cluster skills
                X_faculty = vectorizer_skills.transform(df_faculty['expertise'])

                # Aggregate student skills per cluster
                cluster_skills = final_df.groupby('cluster')['skills'].apply(lambda x: ' '.join(x)).reset_index()

                # Vectorize cluster skills
                X_cluster_skills = vectorizer_skills.transform(cluster_skills['skills'])

                # Calculate similarity between cluster skills and faculty expertise
                similarity_matrix = cosine_similarity(X_cluster_skills, X_faculty)

                # Assign faculty based on highest similarity
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

                # Add assigned faculty to cluster skills DataFrame
                cluster_skills['assigned_faculty_id'] = assigned_faculty

                # Final merge to include assigned faculty in the final DataFrame
                final_data_with_faculty = final_df.merge(cluster_skills[['cluster', 'assigned_faculty_id']], on='cluster', how='left')

                # Prepare final output by excluding faculty skills
                final_output = final_data_with_faculty[['student_id', 'cluster', 'assigned_faculty_id', 'skills', 'interest_to_work']]
                
                # Prepare file for download
                final_file_name = "clustered_students_with_faculty.csv"
                final_output.to_csv(final_file_name, index=False)

                # Provide download link
                st.success("Clustering and faculty assignment completed successfully!")
                st.write("Download the resulting CSV file:")
                with open(final_file_name, "rb") as f:
                    st.download_button(label="Download CSV", data=f, file_name=final_file_name, mime="text/csv")
        except Exception as e:
            st.error(f"Error occurred: {e}")
else:
    st.warning("Please upload both student and faculty data files to proceed.")
