# sql1.py
"""Volume 3: SQL 1 (Introduction).
Caelan Osman
Math 347 Sec. 2
March 21, 2021
"""
import csv
import sqlite3 as sql

# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    #problem 2 populate
    #create rows for Major info and Course info manually
    major_info = [(1, "Math"), (2, "Science"), (3, "Writing"), (4, "Art")]
    course_info = [(1, "Calculus"), (2, "English"), (3, "Pottery"), (4, "History")]

    #pull in student info CSV file and read
    with open(student_info, 'r') as infile:
        stud_info = list(csv.reader(infile))
    #make data the correct type
    stud_info = [tuple(info) for info in stud_info]
    '''
    stud_info = [tuple((int(info[0]), info[1], int(info[2]))) if int(info[2]) != -1
            else tuple((int(info[0]), info[1], None)) for info in stud_info]
    '''

    #pull in student grade CSV file
    with open(student_grades, 'r') as infile:
        grade_info = list(csv.reader(infile))

    #put in correct format
    grade_info = [tuple(grade) for grade in grade_info]
    #grade_info = [tuple((int(grade[0]), int(grade[1]), grade[2])) for grade in grade_info]

    #problem 1 create database
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #drop tables
            cur.execute("DROP TABLE IF EXISTS MajorInfo;")
            cur.execute("DROP TABLE if exists CourseInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentGrades;")
            #create tables
            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT);")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT);")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER);")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT);")
            #populate tables
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", major_info)
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", course_info)
            cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", stud_info)
            cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", grade_info)
            #update table
            cur.execute("UPDATE StudentInfo SET MajorID=NULL where MajorID == -1;")

    finally:
        conn.close()


# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.
    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    with open(data_file, 'r') as infile:
        earthquake_info = list(csv.reader(infile))

    earthquake_info = [tuple(line) for line in earthquake_info]

    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #drop tables
            cur.execute("DROP TABLE IF EXISTS USEarthquakes;")
            #create tables
            cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL);")
            #populate
            cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?, ?);", earthquake_info)
            #delete rows with zero magnitude value
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude == 0")
            #change to null values where Day, Hour, Minute, Second are zero
            cur.execute('UPDATE USEarthquakes SET Day=NULL WHERE Day == 0')
            cur.execute('UPDATE USEarthquakes SET Hour=NULL WHERE Hour == 0')
            cur.execute('UPDATE USEarthquakes SET Minute=NULL WHERE Minute == 0')
            cur.execute('UPDATE USEarthquakes SET Second=NULL WHERE Second == 0')

    finally:
        conn.close()


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """

    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            #select students from database
            cur.execute("SELECT SI.StudentName, CI.CourseName "
                        "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades AS SG "
                        "WHERE SG.Grade IN('A', 'A+') AND "
                        "SG.StudentID == SI.StudentID AND CI.CourseID == SG.CourseID")

            #fetch and return them
            students = list(cur.fetchall())
    finally:
        conn.close()

    return students


# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    raise NotImplementedError("Problem 6 Incomplete")

if __name__ == "__main__":
    #test for problem 1
    '''
    student_db(db_file="students.db", student_info="student_info.csv",
               student_grades="student_grades.csv")
    try:
        with sql.connect("students.db") as conn:
            cur = conn.cursor()

            cur.execute("SELECT * FROM MajorInfo;")
            print([d[0] for d in cur.description])

            cur.execute("select * FROM CourseInfo;")
            print([d[0] for d in cur.description])

            cur.execute("SELECT * FROM StudentInfo;")
            print([d[0] for d in cur.description])

            cur.execute("SELECT * FROM StudentGrades;")
            print([d[0] for d in cur.description])

    finally:
        conn.close()
    #test for problem 2
    '''

    '''
    student_db(db_file="students.db", student_info="student_info.csv",
               student_grades="student_grades.csv")
    try:
        with sql.connect("students.db") as conn:
            cur = conn.cursor()
            for row in cur.execute("SELECT * FROM StudentInfo;"):
                print(row)

    finally:
        conn.close()
    '''

    '''
    earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv")
    try:
        with sql.connect("earthquakes.db") as conn:
            cur = conn.cursor()
            for row in cur.execute("SELECT * FROM USEarthquakes;"):
                print(row)

    finally:
        conn.close()
    '''

    '''
    print(prob5())
    '''
