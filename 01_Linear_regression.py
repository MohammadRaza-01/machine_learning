# ============================================================
# SIMPLE LINEAR REGRESSION
# REAL-WORLD EXAMPLE: SALARY vs EXPERIENCE
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


print("=" * 65)
print("          SALARY PREDICTION USING LINEAR REGRESSION")
print("=" * 65)

try:

    # --------------------------------------------------------
    # STEP 1: Enter training data
    # --------------------------------------------------------

    print("\nTRAINING PHASE")
    print("-" * 65)

    print("""
We will train the model using:

Years of Experience  →  Salary

Example:
Experience = 1
Salary = 25000

You can enter your own data.
""")

    number_of_samples = int(
        input("How many training examples do you want to enter? ")
    )

    if number_of_samples < 2:
        raise ValueError(
            "Please enter at least 2 training examples."
        )

    experience = []
    salary = []

    for i in range(number_of_samples):

        print(f"\nTraining Example {i + 1}")

        exp = float(
            input("Enter years of experience: ")
        )

        sal = float(
            input("Enter salary: ")
        )

        experience.append(exp)
        salary.append(sal)


    # --------------------------------------------------------
    # STEP 2: Convert data into NumPy arrays
    # --------------------------------------------------------

    X = np.array(experience)
    y = np.array(salary)

    # LinearRegression expects X to be 2-dimensional
    X = X.reshape(-1, 1)


    # --------------------------------------------------------
    # STEP 3: Display training data
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("YOUR TRAINING DATA")
    print("=" * 65)

    for exp, sal in zip(
        experience,
        salary
    ):
        print(
            f"Experience: {exp} years"
            f"  →  Salary: ₹{sal:,.2f}"
        )


    # --------------------------------------------------------
    # STEP 4: Create Linear Regression model
    # --------------------------------------------------------

    model = LinearRegression()


    # --------------------------------------------------------
    # STEP 5: Train the model
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TRAINING MODEL")
    print("=" * 65)

    model.fit(X, y)

    print("✅ Model trained successfully!")


    # --------------------------------------------------------
    # STEP 6: Get learned slope and intercept
    # --------------------------------------------------------

    slope = model.coef_[0]
    intercept = model.intercept_

    print("\n" + "=" * 65)
    print("WHAT DID THE MODEL LEARN?")
    print("=" * 65)

    print(f"\nSlope (m)     : {slope:.4f}")
    print(f"Intercept (c) : {intercept:.4f}")

    print("\nLearned equation:")

    print(
        f"Salary = ({slope:.4f} × Experience) "
        f"+ {intercept:.4f}"
    )


    # --------------------------------------------------------
    # STEP 7: Predict salary for training data
    # --------------------------------------------------------

    training_predictions = model.predict(X)


    # --------------------------------------------------------
    # STEP 8: Calculate evaluation metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y,
        training_predictions
    )

    mse = mean_squared_error(
        y,
        training_predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y,
        training_predictions
    )


    # --------------------------------------------------------
    # STEP 9: Display model performance
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE")
    print("=" * 65)

    print(f"\nMAE  (Mean Absolute Error) : ₹{mae:,.2f}")
    print(f"MSE  (Mean Squared Error)  : {mse:,.2f}")
    print(f"RMSE (Root Mean Squared Error): ₹{rmse:,.2f}")
    print(f"R² Score                   : {r2:.4f}")


    # --------------------------------------------------------
    # STEP 10: Actual vs predicted salary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ACTUAL VS PREDICTED SALARY")
    print("=" * 65)

    for exp, actual, predicted in zip(
        experience,
        y,
        training_predictions
    ):

        print(
            f"Experience: {exp} years"
            f" | Actual: ₹{actual:,.2f}"
            f" | Predicted: ₹{predicted:,.2f}"
        )


    # --------------------------------------------------------
    # STEP 11: TEST THE TRAINED MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TEST YOUR TRAINED MODEL")
    print("=" * 65)

    print("""
Now enter new years of experience.

The model will use the relationship
it learned from your training data.

Type 'exit' to stop.
""")

    while True:

        test_input = input(
            "\nEnter years of experience: "
        )

        # Stop testing
        if test_input.lower() == "exit":

            print("\n✅ Testing completed.")
            break

        try:

            new_experience = float(test_input)

            new_data = np.array(
                [[new_experience]]
            )

            predicted_salary = model.predict(
                new_data
            )[0]

            print("\n--------------------------------------")
            print(
                f"Experience       : "
                f"{new_experience} years"
            )

            print(
                f"Predicted Salary : "
                f"₹{predicted_salary:,.2f}"
            )

            print("--------------------------------------")

        except ValueError:

            print(
                "❌ Please enter a valid number "
                "or type 'exit'."
            )


    # --------------------------------------------------------
    # STEP 12: Plot training data and best-fit line
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    # Original training points
    plt.scatter(
        experience,
        salary,
        label="Training Data"
    )

    # Create values for the best-fit line
    x_line = np.linspace(
        min(experience),
        max(experience),
        100
    ).reshape(-1, 1)

    y_line = model.predict(
        x_line
    )

    # Draw best-fit line
    plt.plot(
        x_line,
        y_line,
        label="Best Fit Line"
    )

    plt.xlabel(
        "Years of Experience"
    )

    plt.ylabel(
        "Salary (₹)"
    )

    plt.title(
        "Salary vs Years of Experience"
    )

    plt.legend()
    plt.grid(True)

    plt.show()


except ValueError as error:

    print(
        f"\n❌ Input Error: {error}"
    )

except Exception as error:

    print(
        f"\n❌ Something went wrong: {error}"
    )
