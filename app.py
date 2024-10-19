import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from itertools import chain
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

# Load data from uploaded CSV files
def load_data(student_file, faculty_file):
    df_students = pd.read_csv(student_file)
    df_faculty = pd.read_csv(faculty_file)
    return df_students, df_faculty

# Cluster students based on skills and interests
def cluster_students(df_students):
    vectorizer_skills = TfidfVectorizer()
    X_skills = vectorizer_skills.fit_transform(df_students['skills'])

    vectorizer_interests = TfidfVectorizer()
    X_interests = vectorizer_skills.fit_transform(df_students['interest_to_work'])

    X_students = hstack([X_skills, X_interests])

    num_clusters = (len(df_students) + 2) // 3  
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_students['cluster'] = kmeans.fit_predict(X_students) + 1

    return df_students, num_clusters

# Create balanced student clusters
def create_balanced_clusters(df_students):
    clusters = df_students.groupby('cluster')['student_id'].apply(list).tolist()
    students = list(chain.from_iterable(clusters))
    balanced_clusters = [students[i:i + 3] for i in range(0, len(students), 3)]

    balanced_df = pd.DataFrame([(i, student_id) for i, cluster in enumerate(balanced_clusters) for student_id in cluster],
                               columns=['cluster', 'student_id'])
    return balanced_df

# Cluster faculties and assign unique main and supporting faculties
def cluster_faculty(df_faculty):
    vectorizer_skills = TfidfVectorizer()
    X_faculty = vectorizer_skills.fit_transform(df_faculty['expertise'])

    num_faculty_clusters = (len(df_faculty) + 2) // 3
    kmeans = KMeans(n_clusters=num_faculty_clusters, random_state=42)
    df_faculty['faculty_cluster'] = kmeans.fit_predict(X_faculty)

    return df_faculty, num_faculty_clusters

# Assign main and supporting faculties based on domain constraints
def assign_faculty(df_students, df_faculty, balanced_df):
    vectorizer_skills = TfidfVectorizer()
    X_faculty = vectorizer_skills.fit_transform(df_faculty['expertise'])

    cluster_skills = balanced_df.merge(df_students[['student_id', 'skills']], on='student_id', how='left')
    cluster_skills = cluster_skills.groupby('cluster')['skills'].apply(lambda x: ' '.join(x)).reset_index()

    X_cluster_skills = vectorizer_skills.transform(cluster_skills['skills'])
    similarity_matrix = cosine_similarity(X_cluster_skills, X_faculty)

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

    cluster_skills['assigned_main_faculty_id'] = assigned_faculty

    support_faculty = []
    for cluster_idx in range(len(cluster_skills)):
        main_faculty_id = cluster_skills.iloc[cluster_idx]['assigned_main_faculty_id']
        main_faculty_expertise = df_faculty[df_faculty['faculty_id'] == main_faculty_id]['expertise'].values[0]

        similarity_scores = similarity_matrix[cluster_idx]
        support_ranking = np.argsort(-similarity_scores)

        support_for_cluster = []
        for faculty_id in support_ranking:
            faculty_expertise = df_faculty.iloc[faculty_id]['expertise']

            # Ensure the supporting faculty is from a different domain than the main faculty
            if faculty_expertise != main_faculty_expertise and df_faculty.iloc[faculty_id]['faculty_id'] != main_faculty_id:
                support_for_cluster.append(int(df_faculty.iloc[faculty_id]['faculty_id']))  # Convert to native int


            if len(support_for_cluster) == 2:  # Limit support members to 2 per cluster
                break

        support_faculty.append(support_for_cluster)

    cluster_skills['support_faculty_ids'] = support_faculty
    return cluster_skills

# Main app function for Streamlit
def main():
    st.title("Creative Student Clustering & Faculty Assignment")
    st.markdown("### Upload student and faculty data below to get started!")

    # Upload CSV files
    student_file = st.file_uploader("Upload Student Data (CSV)", type='csv')
    faculty_file = st.file_uploader("Upload Faculty Data (CSV)", type='csv')

    if student_file and faculty_file:
        # Load and process data
        df_students, df_faculty = load_data(student_file, faculty_file)
        df_students, num_clusters = cluster_students(df_students)
        balanced_df = create_balanced_clusters(df_students)
        df_faculty, num_faculty_clusters = cluster_faculty(df_faculty)
        cluster_skills = assign_faculty(df_students, df_faculty, balanced_df)

        # Prepare final output
        final_output = balanced_df.merge(
            cluster_skills[['cluster', 'assigned_main_faculty_id', 'support_faculty_ids']],
            on='cluster', 
            how='left'
        ).merge(
            df_students[['student_id', 'skills', 'interest_to_work', 'cgpa']],
            on='student_id',
            how='left'
        )

        # Display the final results with a creative UI
        st.markdown("### Clustering Results")
        
        # Display each cluster with its faculties in a collapsible section
        for cluster_id in final_output['cluster'].unique():
            with st.expander(f"Cluster {cluster_id + 1}"):
                cluster_data = final_output[final_output['cluster'] == cluster_id]
                st.markdown(f"**Main Faculty ID**: {cluster_data.iloc[0]['assigned_main_faculty_id']}")
                st.markdown(f"**Supporting Faculty IDs**: {', '.join(map(str, cluster_data.iloc[0]['support_faculty_ids']))}")
                st.table(cluster_data[['student_id', 'skills', 'interest_to_work', 'cgpa']])

        # Download button for results
        csv = final_output.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustering Results", csv, "clustered_students_with_faculty.csv", "text/csv")

if __name__ == "__main__":
    main()
