import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from itertools import chain
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

def load_data(student_file, faculty_file):
    # Load student and faculty data
    df_students = pd.read_csv(student_file)
    df_faculty = pd.read_csv(faculty_file)
    return df_students, df_faculty

def cluster_students(df_students):
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

    return df_students, num_clusters

def create_balanced_clusters(df_students):
    # Create balanced clusters
    clusters = df_students.groupby('cluster')['student_id'].apply(list).tolist()
    students = list(chain.from_iterable(clusters))
    balanced_clusters = [students[i:i + 3] for i in range(0, len(students), 3)]

    # Create a DataFrame for final data
    balanced_df = pd.DataFrame([(i, student_id) for i, cluster in enumerate(balanced_clusters) for student_id in cluster],
                               columns=['cluster', 'student_id'])
    return balanced_df

def assign_faculty(df_students, df_faculty, balanced_df):
    # Prepare to assign faculty based on cluster skills
    vectorizer_skills = TfidfVectorizer()
    X_faculty = vectorizer_skills.fit_transform(df_faculty['expertise'])

    # Aggregate student skills per cluster
    cluster_skills = balanced_df.merge(df_students[['student_id', 'skills']], on='student_id', how='left')
    cluster_skills = cluster_skills.groupby('cluster')['skills'].apply(lambda x: ' '.join(x)).reset_index()

    # Vectorize cluster skills
    X_cluster_skills = vectorizer_skills.transform(cluster_skills['skills'])

    # Calculate similarity between cluster skills and faculty expertise
    similarity_matrix = cosine_similarity(X_cluster_skills, X_faculty)

    # Assign faculty based on highest similarity
    assigned_faculty = [-1] * len(cluster_skills)  
    faculty_assigned = set()  

    for cluster_idx in range(len(cluster_skills)):
        similarity_scores = similarity_matrix[cluster_idx]
        faculty_ranking = np.argsort(-similarity_scores)

        for faculty_id in faculty_ranking:
            if faculty_id not in faculty_assigned:
                assigned_faculty[cluster_idx] = df_faculty.iloc[faculty_id]['faculty_id']
                faculty_assigned.add(faculty_id)
                break

    cluster_skills['assigned_faculty_id'] = assigned_faculty
    return cluster_skills

def main():
    st.title("Student Clustering and Faculty Assignment")

    # Upload student and faculty CSV files
    student_file = st.file_uploader("Upload Student Data (CSV)", type='csv')
    faculty_file = st.file_uploader("Upload Faculty Data (CSV)", type='csv')

    if student_file and faculty_file:
        # Load data
        df_students, df_faculty = load_data(student_file, faculty_file)

        # Cluster students
        df_students, num_clusters = cluster_students(df_students)

        # Create balanced clusters
        balanced_df = create_balanced_clusters(df_students)

        # Assign faculty
        cluster_skills = assign_faculty(df_students, df_faculty, balanced_df)

        # Prepare final output by including all required fields
        final_output = balanced_df.merge(
            cluster_skills[['cluster', 'assigned_faculty_id']],
            on='cluster', 
            how='left'
        ).merge(
            df_students[['student_id', 'skills', 'interest_to_work', 'cgpa']],
            on='student_id',
            how='left'
        )

        final_output = final_output[['student_id', 'cluster', 'assigned_faculty_id', 'skills', 'interest_to_work', 'cgpa']]
        
        # Display the results
        st.subheader("Clustering Results")
        st.write(final_output)

        # Download the results as a CSV file
        csv = final_output.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "clustered_students_with_faculty.csv", "text/csv")

if __name__ == "__main__":
    main()
