# Creative Student Clustering & Faculty Assignment
## Link to access the application 
https://student-grouping-k8k8k8jw7pvcnvpbnypuy8.streamlit.app/#clustering-results
## Overview

This project involves clustering students based on their skills and interests, and assigning both main and supporting faculties to each cluster. The application uses a creative UI built with Streamlit, allowing users to upload CSV files for student and faculty data, and outputs clustered results with faculty assignments. Additionally, the results can be downloaded as a CSV file.

## Features

- **Student Clustering**: Students are grouped into clusters based on their skills and interests using the KMeans clustering algorithm.
- **Balanced Clusters**: Each cluster is balanced with up to three students per cluster.
- **Faculty Assignment**: Each cluster is assigned a main faculty and two supporting faculties based on expertise, ensuring that supporting faculties belong to different domains than the main faculty.
- **Creative User Interface**: Built with Streamlit, the interface allows users to upload CSV files, view clustering results with faculties in collapsible sections, and download the final data.
  
## Requirements

The following Python libraries are required to run the code:
  
- `pandas`
- `scikit-learn`
- `itertools`
- `scipy`
- `numpy`
- `streamlit`

You can install these libraries using the following command:
```bash
pip install pandas scikit-learn numpy scipy streamlit
```

## Files

### Input Files

1. **Student Data (CSV)**: This file should include the following columns:
   - `student_id`: Unique ID for each student.
   - `skills`: Skills of the student (text field).
   - `interest_to_work`: Student's areas of interest (text field).
   - `cgpa`: Student's CGPA.
   
2. **Faculty Data (CSV)**: This file should include the following columns:
   - `faculty_id`: Unique ID for each faculty member.
   - `expertise`: Expertise/skills of the faculty member (text field).

### Output Files

- **Clustered Students with Faculty (CSV)**: The output file will include the following columns:
  - `cluster`: Cluster number of the student.
  - `student_id`: Student ID.
  - `skills`: Skills of the student.
  - `interest_to_work`: Interests of the student.
  - `cgpa`: CGPA of the student.
  - `assigned_main_faculty_id`: ID of the main faculty assigned to the cluster.
  - `support_faculty_ids`: IDs of the supporting faculties assigned to the cluster.

## Code Structure

### 1. Load Data
The `load_data` function loads student and faculty data from CSV files:
```python
def load_data(student_file, faculty_file):
    df_students = pd.read_csv(student_file)
    df_faculty = pd.read_csv(faculty_file)
    return df_students, df_faculty
```

### 2. Student Clustering
The `cluster_students` function clusters students based on their skills and interests using TF-IDF vectorization and KMeans clustering:
```python
def cluster_students(df_students):
    ...
    df_students['cluster'] = kmeans.fit_predict(X_students)
    return df_students, num_clusters
```

### 3. Create Balanced Clusters
The `create_balanced_clusters` function ensures each cluster has a maximum of three students:
```python
def create_balanced_clusters(df_students):
    ...
    return balanced_df
```

### 4. Faculty Clustering
The `cluster_faculty` function clusters faculty members based on their expertise using TF-IDF vectorization and KMeans clustering:
```python
def cluster_faculty(df_faculty):
    ...
    return df_faculty, num_faculty_clusters
```

### 5. Faculty Assignment
The `assign_faculty` function assigns both main and supporting faculties to each cluster based on cosine similarity between student skills and faculty expertise. Supporting faculties are ensured to be from different domains:
```python
def assign_faculty(df_students, df_faculty, balanced_df):
    ...
    return cluster_skills
```

### 6. Streamlit User Interface
The `main` function provides an interactive Streamlit UI where users can upload CSV files, view clustering results, and download the output:
```python
def main():
    ...
    if student_file and faculty_file:
        ...
        st.download_button("Download Clustering Results", csv, "clustered_students_with_faculty.csv", "text/csv")
```

## How to Run

1. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. In the browser, upload the student and faculty CSV files.

4. View the results of the clustering with assigned faculties.

5. Download the final clustering results as a CSV file.

## Notes

- Make sure the student and faculty CSV files are correctly formatted.
- Clustering starts at 1, ensuring no 0-indexed clusters.
- Faculty members are assigned based on cosine similarity, and supporting faculties are from different domains than the main faculty.
