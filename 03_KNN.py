# ============================================================
# KNN STUDENT PASS / FAIL PREDICTION
# ============================================================
#
# Features:
# 1. Study Hours / Day
# 2. Practice Tests
# 3. Attendance %
# 4. Previous Exam Score %
# 5. Sleep Hours / Day
#
# Target:
# Pass / Fail
#
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 2. TAKE TRAINING DATA FROM USER
# ------------------------------------------------------------

print("=" * 60)
print("       KNN STUDENT PASS / FAIL PREDICTION")
print("=" * 60)

print("\nTraining Data")
print("-" * 60)

print("""
For every student enter:

1. Study Hours / Day
2. Practice Tests
3. Attendance %
4. Previous Exam Score %
5. Sleep Hours / Day
6. Result (Pass / Fail)
""")


# Number of training students
while True:
    try:
        n = int(input("Enter number of training students: "))

        if n < 3:
            print("Please enter at least 3 students.")
        else:
            break

    except ValueError:
        print("Please enter a valid integer.")


X = []
y = []


# ------------------------------------------------------------
# 3. ENTER TRAINING DATA
# ------------------------------------------------------------

for i in range(n):

    print("\n" + "-" * 60)
    print(f"Student {i + 1}")
    print("-" * 60)

    while True:
        try:
            study_hours = float(input("Study Hours / Day: "))

            if study_hours < 0:
                print("Study hours cannot be negative.")
                continue

            break

        except ValueError:
            print("Enter a valid number.")


    while True:
        try:
            practice_tests = float(input("Practice Tests: "))

            if practice_tests < 0:
                print("Practice tests cannot be negative.")
                continue

            break

        except ValueError:
            print("Enter a valid number.")


    while True:
        try:
            attendance = float(input("Attendance %: "))

            if attendance < 0 or attendance > 100:
                print("Attendance must be between 0 and 100.")
                continue

            break

        except ValueError:
            print("Enter a valid number.")


    while True:
        try:
            previous_score = float(
                input("Previous Exam Score %: ")
            )

            if previous_score < 0 or previous_score > 100:
                print("Score must be between 0 and 100.")
                continue

            break

        except ValueError:
            print("Enter a valid number.")


    while True:
        try:
            sleep_hours = float(input("Sleep Hours / Day: "))

            if sleep_hours < 0 or sleep_hours > 24:
                print("Sleep hours must be between 0 and 24.")
                continue

            break

        except ValueError:
            print("Enter a valid number.")


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    while True:

        result = input("Result (Pass/Fail): ").strip().capitalize()

        if result in ["Pass", "Fail"]:
            break

        print("Please enter only Pass or Fail.")


    # Store features
    X.append([
        study_hours,
        practice_tests,
        attendance,
        previous_score,
        sleep_hours
    ])

    # Store target
    y.append(result)


# ------------------------------------------------------------
# 4. CONVERT DATA INTO NUMPY ARRAYS
# ------------------------------------------------------------

X = np.array(X)
y = np.array(y)


print("\n" + "=" * 60)
print("TRAINING DATA COLLECTED")
print("=" * 60)

print(f"Total students: {len(X)}")


# ------------------------------------------------------------
# 5. CHOOSE K
# ------------------------------------------------------------

while True:

    try:

        k = int(input("\nEnter value of K: "))

        if k <= 0:
            print("K must be greater than 0.")
            continue

        if k > len(X):
            print("K cannot be greater than number of training students.")
            continue

        break

    except ValueError:
        print("Enter a valid integer.")


# ------------------------------------------------------------
# 6. FEATURE SCALING
# ------------------------------------------------------------
#
# KNN uses distance.
#
# Our features have different ranges:
#
# Attendance       -> 0 to 100
# Previous Score   -> 0 to 100
# Study Hours      -> around 0 to 15
# Sleep Hours      -> around 0 to 12
#
# Therefore we scale them before KNN.
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------
# 7. CREATE KNN MODEL
# ------------------------------------------------------------

model = KNeighborsClassifier(
    n_neighbors=k,
    metric="euclidean"
)


# ------------------------------------------------------------
# 8. TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_scaled, y)


print("\n" + "=" * 60)
print("           MODEL TRAINING COMPLETED")
print("=" * 60)

print(f"Number of training students : {len(X)}")
print(f"K value                     : {k}")
print("Distance metric             : Euclidean")


# ------------------------------------------------------------
# 9. DISPLAY TRAINING DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              TRAINING DATA")
print("=" * 60)

print(
    f"{'No.':<5}"
    f"{'Study':<10}"
    f"{'Tests':<10}"
    f"{'Attend.':<10}"
    f"{'Score':<10}"
    f"{'Sleep':<10}"
    f"{'Result':<10}"
)

print("-" * 65)

for i in range(len(X)):

    print(
        f"{i + 1:<5}"
        f"{X[i][0]:<10.1f}"
        f"{X[i][1]:<10.1f}"
        f"{X[i][2]:<10.1f}"
        f"{X[i][3]:<10.1f}"
        f"{X[i][4]:<10.1f}"
        f"{y[i]:<10}"
    )


# ------------------------------------------------------------
# 10. TESTING LOOP
# ------------------------------------------------------------

while True:

    print("\n" + "=" * 60)
    print("               TEST NEW STUDENT")
    print("=" * 60)

    print("""
Enter the new student's information.
Type 'exit' when asked for Study Hours if
you want to stop testing.
""")


    # --------------------------------------------------------
    # Study Hours
    # --------------------------------------------------------

    while True:

        study_input = input("Study Hours / Day: ").strip()

        if study_input.lower() == "exit":
            break

        try:

            study_hours = float(study_input)

            if study_hours < 0:
                print("Study hours cannot be negative.")
                continue

            break

        except ValueError:
            print("Enter a valid number.")


    # Exit testing
    if study_input.lower() == "exit":
        break


    # --------------------------------------------------------
    # Other test features
    # --------------------------------------------------------

    while True:
        try:

            practice_tests = float(
                input("Practice Tests: ")
            )

            if practice_tests < 0:
                print("Practice tests cannot be negative.")
                continue

            break

        except ValueError:
            print("Enter a valid number.")


    while True:
        try:

            attendance = float(
                input("Attendance %: ")
            )

            if attendance < 0 or attendance > 100:
                print("Attendance must be between 0 and 100.")
                continue

            break

        except ValueError:
            print("Enter a valid number.")


    while True:
        try:

            previous_score = float(
                input("Previous Exam Score %: ")
            )

            if previous_score < 0 or previous_score > 100:
                print("Score must be between 0 and 100.")
                continue

            break

        except ValueError:
            print("Enter a valid number.")


    while True:
        try:

            sleep_hours = float(
                input("Sleep Hours / Day: ")
            )

            if sleep_hours < 0 or sleep_hours > 24:
                print("Sleep hours must be between 0 and 24.")

                continue

            break

        except ValueError:
            print("Enter a valid number.")


    # --------------------------------------------------------
    # Create test student
    # --------------------------------------------------------

    new_student = np.array([[
        study_hours,
        practice_tests,
        attendance,
        previous_score,
        sleep_hours
    ]])


    # --------------------------------------------------------
    # SCALE TEST DATA
    # --------------------------------------------------------

    new_student_scaled = scaler.transform(new_student)


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(new_student_scaled)


    # --------------------------------------------------------
    # GET PROBABILITY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        new_student_scaled
    )[0]


    # --------------------------------------------------------
    # GET NEAREST NEIGHBORS
    # --------------------------------------------------------

    distances, indices = model.kneighbors(
        new_student_scaled
    )


    # --------------------------------------------------------
    # DISPLAY INPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("             STUDENT INFORMATION")
    print("=" * 60)

    print(f"Study Hours / Day       : {study_hours}")
    print(f"Practice Tests          : {practice_tests}")
    print(f"Attendance              : {attendance}%")
    print(f"Previous Exam Score     : {previous_score}%")
    print(f"Sleep Hours / Day       : {sleep_hours}")


    # --------------------------------------------------------
    # DISPLAY NEAREST STUDENTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"       {k} NEAREST STUDENTS")
    print("=" * 60)

    print(
        f"{'Student':<10}"
        f"{'Distance':<15}"
        f"{'Result':<10}"
    )

    print("-" * 40)

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        print(
            f"{index + 1:<10}"
            f"{distance:<15.4f}"
            f"{y[index]:<10}"
        )


    # --------------------------------------------------------
    # DISPLAY VOTING
    # --------------------------------------------------------

    nearest_results = y[indices[0]]

    pass_votes = np.sum(
        nearest_results == "Pass"
    )

    fail_votes = np.sum(
        nearest_results == "Fail"
    )


    print("\n" + "=" * 60)
    print("                 KNN VOTING")
    print("=" * 60)

    print(f"Pass votes : {pass_votes}")
    print(f"Fail votes : {fail_votes}")


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("                 FINAL PREDICTION")
    print("=" * 60)

    print(f"\nPredicted Result: {prediction[0].upper()}")


    # --------------------------------------------------------
    # PREDICTION PROBABILITY
    # --------------------------------------------------------

    print("\nPrediction Probabilities:")

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):

        print(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )


    print("\n" + "=" * 60)


# ------------------------------------------------------------
# 11. END PROGRAM
# ------------------------------------------------------------

print("\nTesting finished.")
print("Thank you for using the KNN Student Predictor!")
