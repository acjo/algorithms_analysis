# solutions.py
"""Volume 3: SQL 2.
Caelan Osman
Math 347 Sec. 2
March 30, 2021
"""

import csv
import sqlite3 as sql
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """

    #open database
    with sql.connect(db_file) as conn:
        #grab cursor and execute finding student names
        cur = conn.cursor()

        cur.execute("SELECT SI.StudentName "
                    "FROM StudentInfo AS SI INNER JOIN StudentGrades AS SG "
                    "ON SI.StudentID == SG.StudentID "
                    "WHERE SG.Grade == 'B';")
        #fetchall students
        students = cur.fetchall()
    #close the database
    conn.close()
    #return the list as desired
    return [str(student[0]) for student in students]


# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    #open database
    with sql.connect(db_file) as conn:
        #grab the cursor
        cur = conn.cursor()
        #grab what we need from tables
        cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade "
                    "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo as MI "
                    "ON SI.MajorID == MI.MajorID "
                    "INNER JOIN StudentGrades as SG "
                    "ON SG.CourseID == 1 "
                    "WHERE SG.StudentID == SI.StudentID ")
        #fetch the list
        students = cur.fetchall()
    #close database
    conn.close()
    #return the list
    return students

# Problem 3
def prob3(db_file="students.db"):
    """Query the database for the list of the names of courses that have at
    least 5 students enrolled in them.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        ((list): a list of strings, each of which is a course name.
    """
    #open database
    with sql.connect(db_file) as conn:
        #get cursor
        cur = conn.cursor()
        #grab desired courses
        cur.execute("SELECT CI.CourseName "
                    "FROM StudentGrades AS SG INNER JOIN CourseInfo AS CI "
                    "ON CI.CourseID == SG.CourseID "
                    "GROUP BY SG.CourseID "
                    "HAVING COUNT(*) >= 5;")
        #grab the courses
        courses = cur.fetchall()

    #close database
    conn.close()
    #return list of strings
    return [course[0] for course in courses]


# Problem 4
def prob4(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    #open database
    with sql.connect(db_file) as conn:
        #get cursor
        cur = conn.cursor()
        #grab desired courses
        cur.execute("SELECT MI.MajorName, COUNT(*) as num_students "
                    "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                    "ON MI.MajorID == SI.MajorID "
                    "GROUP BY SI.MajorID "
                    "ORDER BY num_students ASC, MI.MajorName ASC;")
        #grab the majors
        majors = cur.fetchall()

    #close database
    conn.close()
    #return major list
    return majors

# Problem 5
def prob5(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, MajorName) where
    the last name of the specified student begins with the letter C.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    #open database
    with sql.connect(db_file) as conn:
        #get cursor
        cur = conn.cursor()
        #grab desired courses
        cur.execute("SELECT SI.StudentName, MI.MajorName "
                    "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                    "ON MI.MajorID == SI.MajorID "
                    "WHERE SI.StudentName LIKE '% C%';")
        #grab the names
        names = cur.fetchall()

    #close database
    conn.close()
    #return name/major list
    return names


# Problem 6
def prob6(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    #open database
    with sql.connect(db_file) as conn:
        #get cursor
        cur = conn.cursor()
        #grab desired courses
        cur.execute("SELECT Name, COUNT(*) , AVG(GPA) as avg_GPA "
                    "FROM ( "
                    "SELECT SI.StudentName AS Name, SG.CourseID as Course, CASE "
                    "WHEN Grade in ('A', 'A+') THEN 4.0 "
                    "WHEN Grade == 'A-' THEN 3.7 "
                    "WHEN Grade == 'B+' THEN 3.4 "
                    "WHEN Grade == 'B' THEN 3.0 "
                    "WHEN Grade == 'B-' THEN 2.7 "
                    "WHEN Grade == 'C+' THEN 2.4 "
                    "WHEN Grade == 'C' THEN 2.0 "
                    "WHEN Grade == 'C-' THEN 1.7 "
                    "WHEN Grade == 'D+' THEN 1.4 "
                    "WHEN Grade == 'D' THEN 1.0 "
                    "WHEN Grade == 'D-' THEN 0.7 END AS GPA "
                    "FROM StudentGrades AS SG LEFT OUTER JOIN StudentInfo AS SI "
                    "ON SG.StudentID == SI.StudentID) "
                    "GROUP BY Name "
                    "ORDER BY avg_GPA DESC;")
        #grab the names
        GPA = cur.fetchall()

    #close database
    conn.close()
    #return name/major list
    return GPA


if __name__ == "__main__":

    '''
    #prob1
    #print(prob1())

    #prob2
    #print(prob2())

    #prob3
    #print(prob3())

    #prob4
    #print(prob4())

    #prob5
    #print(prob5())

    #prob6
    #print(prob6())
    '''
    pass
