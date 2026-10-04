# ================================================================
# MULTIPLE LINEAR REGRESSION
# PROBLEM: PREDICT DAILY INTERNET DATA USAGE
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("=" * 75)
print("       DAILY INTERNET DATA USAGE PREDICTION")
print("             MULTIPLE LINEAR REGRESSION")
print("=" * 75)


try:

    # ============================================================
    # STEP 1: Define Features
    # ============================================================

    features = [
        "Mobile Usage Hours",
        "Instagram Hours",
        "YouTube Hours",
        "Chrome Hours",
        "Gaming Hours",
        "Movies/Streaming Hours",
        "WhatsApp Hours"
    ]

    print("\nMODEL FEATURES")
    print("-" * 75)

    for i, feature in enumerate(features, start=1):
        print(f"{i}. {feature}")

    print("\nTarget:")
    print("Daily Internet Data Usage (GB)")


    # ============================================================
    # STEP 2: Ask Number of Training Examples
    # ============================================================

    print("\n" + "=" * 75)
    print("ENTER TRAINING DATA")
    print("=" * 75)

    print("""
For every training example, enter:

1. Total mobile usage hours
2. Instagram usage hours
3. YouTube usage hours
4. Chrome usage hours
5. Gaming hours
6. Movies/Streaming hours
7. WhatsApp usage hours
8. Actual daily internet usage in GB

Example:

Mobile usage       = 8
Instagram          = 1.5
YouTube            = 2
Chrome             = 1
Gaming             = 1
Movies             = 1
WhatsApp           = 1
Data usage         = 5.5
""")


    number_of_samples = int(
        input("\nHow many training examples do you want to enter? ")
    )

    if number_of_samples < 3:
        raise ValueError(
            "Please enter at least 3 training examples."
        )


    # ============================================================
    # STEP 3: Collect Training Data
    # ============================================================

    X_data = []
    y_data = []

    for i in range(number_of_samples):

        print("\n" + "-" * 75)
        print(f"TRAINING EXAMPLE {i + 1}")
        print("-" * 75)

        mobile_usage = float(
            input("Total mobile usage (hours): ")
        )

        instagram = float(
            input("Instagram usage (hours): ")
        )

        youtube = float(
            input("YouTube usage (hours): ")
        )

        chrome = float(
            input("Chrome usage (hours): ")
        )

        gaming = float(
            input("Gaming usage (hours): ")
        )

        movies = float(
            input("Movies/Streaming usage (hours): ")
        )

        whatsapp = float(
            input("WhatsApp usage (hours): ")
        )

        data_usage = float(
            input("Actual daily internet usage (GB): ")
        )


        # Store features
        X_data.append([
            mobile_usage,
            instagram,
            youtube,
            chrome,
            gaming,
            movies,
            whatsapp
        ])

        # Store target
        y_data.append(data_usage)


    # ============================================================
    # STEP 4: Convert Data to DataFrame
    # ============================================================

    df = pd.DataFrame(
        X_data,
        columns=features
    )

    df["Daily Data Usage (GB)"] = y_data


    # ============================================================
    # STEP 5: Display Training Dataset
    # ============================================================

    print("\n" + "=" * 75)
    print("YOUR TRAINING DATA")
    print("=" * 75)

    print(
        df.to_string(index=False)
    )


    # ============================================================
    # STEP 6: Separate Features and Target
    # ============================================================

    X = df[features]

    y = df["Daily Data Usage (GB)"]


    # ============================================================
    # STEP 7: Create Multiple Linear Regression Model
    # ============================================================

    model = LinearRegression()


    # ============================================================
    # STEP 8: Train Model
    # ============================================================

    print("\n" + "=" * 75)
    print("TRAINING MULTIPLE LINEAR REGRESSION MODEL")
    print("=" * 75)

    model.fit(X, y)

    print("\n✅ Model trained successfully!")


    # ============================================================
    # STEP 9: Show Learned Coefficients
    # ============================================================

    print("\n" + "=" * 75)
    print("WHAT DID THE MODEL LEARN?")
    print("=" * 75)

    print(
        f"\nIntercept = {model.intercept_:.4f}"
    )

    print("\nFeature Coefficients:")

    for feature, coefficient in zip(
        features,
        model.coef_
    ):

        print(
            f"{feature:<30} : {coefficient:.4f}"
        )


    # ============================================================
    # STEP 10: Display Mathematical Equation
    # ============================================================

    print("\n" + "=" * 75)
    print("LEARNED MULTIPLE LINEAR REGRESSION EQUATION")
    print("=" * 75)

    equation = (
        f"Data Usage = {model.intercept_:.4f}"
    )

    for feature, coefficient in zip(
        features,
        model.coef_
    ):

        equation += (
            f" + ({coefficient:.4f} × {feature})"
        )

    print("\n")
    print(equation)


    # ============================================================
    # STEP 11: Make Predictions on Training Data
    # ============================================================

    training_predictions = model.predict(X)


    # ============================================================
    # STEP 12: Calculate Model Performance
    # ============================================================

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


    # ============================================================
    # STEP 13: Display Evaluation
    # ============================================================

    print("\n" + "=" * 75)
    print("MODEL PERFORMANCE")
    print("=" * 75)

    print(
        f"\nMAE  : {mae:.4f} GB"
    )

    print(
        f"MSE  : {mse:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f} GB"
    )

    print(
        f"R²   : {r2:.4f}"
    )


    # ============================================================
    # STEP 14: Actual vs Predicted
    # ============================================================

    print("\n" + "=" * 75)
    print("ACTUAL VS PREDICTED DATA USAGE")
    print("=" * 75)

    for i, (actual, predicted) in enumerate(
        zip(y, training_predictions),
        start=1
    ):

        print(
            f"Example {i:<3}"
            f" | Actual: {actual:.2f} GB"
            f" | Predicted: {predicted:.2f} GB"
        )


    # ============================================================
    # STEP 15: TEST THE TRAINED MODEL
    # ============================================================

    print("\n" + "=" * 75)
    print("TEST YOUR TRAINED MODEL")
    print("=" * 75)

    print("""
Now enter a NEW user's daily usage information.

The model will predict the daily internet data consumption.

Type 'exit' when you want to stop testing.
""")


    while True:

        print("\n" + "-" * 75)

        test_mobile = input(
            "Total mobile usage hours: "
        )

        if test_mobile.lower() == "exit":
            break

        try:

            test_mobile = float(
                test_mobile
            )

            test_instagram = float(
                input("Instagram hours: ")
            )

            test_youtube = float(
                input("YouTube hours: ")
            )

            test_chrome = float(
                input("Chrome hours: ")
            )

            test_gaming = float(
                input("Gaming hours: ")
            )

            test_movies = float(
                input("Movies/Streaming hours: ")
            )

            test_whatsapp = float(
                input("WhatsApp hours: ")
            )


            # Create test input
            test_data = pd.DataFrame(
                [[
                    test_mobile,
                    test_instagram,
                    test_youtube,
                    test_chrome,
                    test_gaming,
                    test_movies,
                    test_whatsapp
                ]],
                columns=features
            )


            # Make prediction
            prediction = model.predict(
                test_data
            )[0]


            print("\n" + "=" * 75)
            print("PREDICTION")
            print("=" * 75)

            print(
                f"\nPredicted Daily Internet Usage"
                f" = {prediction:.2f} GB"
            )

            print("=" * 75)


        except ValueError:

            print(
                "\n❌ Please enter valid numerical values."
            )


    # ============================================================
    # STEP 16: Feature Coefficient Visualization
    # ============================================================

    plt.figure(figsize=(12, 6))

    plt.bar(
        features,
        model.coef_
    )

    plt.xlabel(
        "Features"
    )

    plt.ylabel(
        "Coefficient"
    )

    plt.title(
        "Multiple Linear Regression - Feature Coefficients"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.show()


except ValueError as error:

    print(
        f"\n❌ Input Error: {error}"
    )

except Exception as error:

    print(
        f"\n❌ Something went wrong: {error}"
    )
