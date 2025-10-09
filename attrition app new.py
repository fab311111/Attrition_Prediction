import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib # Import joblib to load the scaler and model if saved externally

# Load the original dataset
# Make sure to adjust the path if your file is not in the same directory as the app.py file
try:
    df_original = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
except FileNotFoundError:
    st.error("Error: WA_Fn-UseC_-HR-Employee-Attrition.csv not found. Please make sure the file is in the correct directory.")
    st.stop() # Stop the app if the data file is not found

# Load the pre-trained model and scaler
# Assuming you have saved your trained model and scaler using joblib or pickle
# Replace 'best_rf_model.pkl' and 'scaler.pkl' with your actual filenames
try:
    # If you trained the model and scaler in the same notebook session and are running this script
    # in that environment, the variables might be available directly.
    # However, for a standalone app, it's best to save and load them.
    # Example of saving:
    # import joblib
    # joblib.dump(best_rf_model, 'best_rf_model.pkl')
    # joblib.dump(scaler, 'scaler.pkl')

    # For this example, we assume the variables are available if run within the notebook context
    # If running as a standalone app, uncomment the joblib loading lines below
    # best_rf_model = joblib.load('best_rf_model.pkl')
    # scaler = joblib.load('scaler.pkl')
    pass # Assuming model and scaler are available in the environment

except FileNotFoundError:
    st.error("Error: Trained model or scaler file not found. Please make sure 'best_rf_model' and 'scaler' are available in the environment or load them from files.")
    st.stop()


# Set the title of the Streamlit application
st.title('Employee Attrition Risk Predictor')

# Add a text input field for the employee number
employee_id_input = st.text_input('Enter Employee Number:')

# Ensure employee_id is not empty before proceeding
if employee_id_input:
    try:
        employee_id = int(employee_id_input)

        # 1. Filter df_original to get the row corresponding to the input employee_id
        # Assuming 'EmployeeNumber' is the column name for employee IDs
        employee_data = df_original[df_original['EmployeeNumber'] == employee_id].copy()

        if employee_data.empty:
            st.warning(f"Employee number {employee_id} not found.")
        else:
            # 2. Drop the same irrelevant columns
            columns_to_drop = [
                'EmployeeCount',
                'StandardHours',
                'Over18',
                'EmployeeNumber'
            ]
            # Keep the original employee data for risk factor analysis later
            employee_original_data = employee_data.iloc[0].copy()

            employee_data = employee_data.drop(columns=columns_to_drop)

            # Separate target variable 'Attrition' before one-hot encoding if it exists
            if 'Attrition' in employee_data.columns:
                 employee_data = employee_data.drop(columns=['Attrition'])


            # 3. Apply one-hot encoding, ensuring all categories from training data are represented
            # Get categorical columns from the original data (before dropping irrelevant ones)
            # Re-calculate categorical columns from df_original excluding dropped columns
            original_categorical_cols = df_original.select_dtypes(include=['object']).columns.tolist()
            original_categorical_cols = [col for col in original_categorical_cols if col not in columns_to_drop]


            # Apply one-hot encoding
            employee_data_encoded = pd.get_dummies(employee_data, columns=original_categorical_cols, drop_first=True)

            # 4. Ensure columns are in the same order as the training data and add missing columns
            # You'll need access to the columns of the training data (X_train)
            # Assuming X_train.columns is available from your training script
            # If not, you would need to save and load the list of column names
            try:
                train_cols = X_train.columns # Access the global X_train if available
            except NameError:
                 st.error("Error: Training columns (X_train.columns) not found. Please ensure X_train is available or load column names from a file.")
                 st.stop()

            # Add missing columns to the employee data and fill with 0
            missing_cols = set(train_cols) - set(employee_data_encoded.columns)
            for c in missing_cols:
                employee_data_encoded[c] = 0

            # Ensure the order of columns is the same as train_cols
            employee_data_processed = employee_data_encoded[train_cols]

            # 5. Apply the same standard scaling to numerical features
            # Identify numerical columns from the original data (before dropping/encoding)
            original_numerical_cols = df_original.select_dtypes(include=[np.number]).columns.drop(['EmployeeCount', 'StandardHours', 'EmployeeNumber']) # Exclude dropped and ID
            numerical_features_after_encoding = employee_data_processed.columns[employee_data_processed.columns.isin(original_numerical_cols)]


            # Apply the pre-fitted scaler
            # Assuming 'scaler' is available from your training script or loaded
            try:
                 employee_data_processed[numerical_features_after_encoding] = scaler.transform(employee_data_processed[numerical_features_after_encoding])
            except NameError:
                 st.error("Error: Scaler object not found. Please ensure 'scaler' is available or load it from a file.")
                 st.stop()


            # Store the preprocessed single employee data
            single_employee_preprocessed = employee_data_processed

            st.write("Employee data preprocessed successfully.")

            # ==============================================================================
            # Predict attrition risk
            # ==============================================================================
            st.header("Attrition Risk Prediction")

            try:
                # Predict the probability of attrition
                # Assuming 'best_rf_model' is available from your training script or loaded
                try:
                    attrition_probability = best_rf_model.predict_proba(single_employee_preprocessed)[:, 1][0]
                except NameError:
                     st.error("Error: Trained model object not found. Please ensure 'best_rf_model' is available or load it from a file.")
                     st.stop()


                # Define thresholds for risk levels
                low_risk_threshold = 0.3
                medium_risk_threshold = 0.6

                # Assign the attrition risk level
                if attrition_probability < low_risk_threshold:
                    risk_level = 'Low'
                    st.info(f"Predicted Attrition Risk: **{risk_level}** (Probability: {attrition_probability:.2f})")
                elif attrition_probability < medium_risk_threshold:
                    risk_level = 'Medium'
                    st.warning(f"Predicted Attrition Risk: **{risk_level}** (Probability: {attrition_probability:.2f})")
                else:
                    risk_level = 'High'
                    st.error(f"Predicted Attrition Risk: **{risk_level}** (Probability: {attrition_probability:.2f})")

                # Store risk_level in session state for potential future use (optional in this combined script)
                st.session_state['risk_level'] = risk_level


            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


            # ==============================================================================
            # Identify risk factors
            # ==============================================================================
            st.subheader("Analysis of Risk Factors:")

            try:
                # Define key features to examine for risk factors
                # Use original column names for comparison
                key_features_to_examine = [
                    'MonthlyIncome',
                    'Age',
                    'YearsAtCompany',
                    'OverTime',
                    'JobSatisfaction',
                    'EnvironmentSatisfaction',
                    'WorkLifeBalance',
                    'MaritalStatus'
                ]

                # Initialize risk_factors to an empty list
                risk_factors = []

                # Calculate typical values for attriters for the key features from df_original
                # Re-calculate high_risk_employees based on df_original and 'Attrition' == 'Yes'
                high_risk_employees_original = df_original[df_original['Attrition'] == 'Yes'].copy()


                for feature in key_features_to_examine:
                    if feature in employee_original_data.index and feature in high_risk_employees_original.columns:
                        employee_value = employee_original_data[feature]

                        if high_risk_employees_original[feature].dtype in ['int64', 'float64']:
                            # Numerical feature
                            typical_attriter_mean = high_risk_employees_original[feature].mean()
                            typical_attriter_std = high_risk_employees_original[feature].std()

                            # Simple check: if employee value is within a certain range of the attriter mean
                            # Using 0.5 standard deviations as a more sensitive threshold
                            if abs(employee_value - typical_attriter_mean) <= 0.5 * typical_attriter_std:
                                risk_factors.append({
                                    'feature': feature,
                                    'employee_value': employee_value,
                                    'typical_attriter_value': f"Around {typical_attriter_mean:.2f} (std: {typical_attriter_std:.2f})",
                                    'comment': f"Employee's {feature} ({employee_value}) is close to the average for attriting employees."
                                })
                        else:
                            # Categorical feature
                            attriter_value_counts = high_risk_employees_original[feature].value_counts(normalize=True)
                            if employee_value in attriter_value_counts.index:
                                attriter_percentage = attriter_value_counts.loc[employee_value] * 100

                                # Simple check: if the employee's category is prevalent among attriters (e.g., > 25%)
                                if attriter_percentage > 25: # Increased threshold slightly
                                      risk_factors.append({
                                          'feature': feature,
                                          'employee_value': employee_value,
                                          'typical_attriter_value': f"{employee_value} is found in {attriter_percentage:.2f}% of attriters",
                                          'comment': f"Employee's {feature} ({employee_value}) is common among attriting employees."
                                      })


                # Display the identified risk factors
                st.subheader("Identified Risk Factors")
                if risk_factors:
                    st.write("Based on the analysis, the following factors might contribute to the employee's attrition risk:")
                    for factor in risk_factors:
                        st.write(f"- **{factor['feature']}**: Employee value is '{factor['employee_value']}'. {factor['comment']}")
                else:
                    st.info("No specific risk factors identified based on the selected key features compared to typical attriters.")

                # Store the risk_factors for the next step
                st.session_state['risk_factors'] = risk_factors

            except Exception as e:
                st.error(f"An error occurred during risk factor analysis: {e}")
                st.session_state['risk_factors'] = [] # Ensure risk_factors is defined


            # ==============================================================================
            # Suggest mitigation measures
            # ==============================================================================
            st.subheader("Suggested Mitigation Measures:")

            if 'risk_factors' in st.session_state and st.session_state['risk_factors']:
                for factor in st.session_state['risk_factors']:
                    feature = factor['feature']
                    employee_value = factor['employee_value']

                    suggestion = f"- **{feature}**: "

                    if feature == 'MonthlyIncome':
                        suggestion += f"Review compensation and consider salary adjustments or bonuses to ensure competitiveness, especially since the employee's income ({employee_value}) is close to the average for attriting employees."
                    elif feature == 'Age':
                        suggestion += f"Understand the specific needs and career aspirations of employees in the {employee_value} age group. Offer relevant development opportunities or mentorship programs."
                    elif feature == 'YearsAtCompany':
                         suggestion += f"For employees with {employee_value} years at the company, focus on career pathing and recognition. Long-tenured employees who attrite may feel stagnant."
                    elif feature == 'OverTime':
                         suggestion += f"({employee_value}) Implement policies to reduce mandatory overtime and promote a healthier work-life balance. High overtime is a significant factor for employees leaving."
                    elif feature == 'JobSatisfaction':
                         suggestion += f"Investigate the reasons behind the employee's job satisfaction level ({employee_value}). Conduct stay interviews or surveys to identify areas for improvement in their role or responsibilities."
                    elif feature == 'EnvironmentSatisfaction':
                         suggestion += f"Assess and improve the work environment based on the employee's satisfaction level ({employee_value}). This could involve physical workspace, team dynamics, or resources."
                    elif feature == 'WorkLifeBalance':
                         suggestion += f"Support better work-life balance ({employee_value}) through flexible work arrangements, managing workload, and encouraging taking time off."
                    elif feature == 'MaritalStatus':
                         if employee_value == 'Single':
                             suggestion += f"({employee_value}) Understand if being single presents unique challenges or needs within the company culture or benefits structure, as single employees show a higher attrition rate."
                         else: # This case might not be reached with the current risk factor logic but is kept for completeness
                             suggestion += f"({employee_value}) Understand the specific needs related to their marital status."
                    else:
                        suggestion += "No specific mitigation suggestion available for this factor based on current analysis."

                    st.write(suggestion)

            else:
                st.info("No specific mitigation measures suggested based on the current analysis of key factors.")


    except ValueError:
        st.error("Please enter a valid integer for the employee number.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")