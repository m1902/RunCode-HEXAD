import pandas as pd

def load_data(submission_path: str, student_path: str):
    submissions = pd.read_csv(submission_path, sep=";", parse_dates=["upload_date"])
    students = pd.read_csv(student_path, sep=";")
    return submissions, students
