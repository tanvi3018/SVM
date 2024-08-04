# Employee Attrition using SVM


## Purpose

The Employee Attrition using SVM project aims to predict employee attrition using Support Vector Machines (SVM), a powerful classification algorithm. This project explores various features such as employee demographics, job satisfaction, work-life balance, and more to build a model that can classify whether an employee is likely to leave the organization. The goal is to help HR departments proactively address issues that might lead to higher turnover rates.

## How to Run

To run the project, follow these steps:

    Clone the Repository:

    sh

git clone https://github.com/yourusername/Employee_Attrition_using_SVM.git
cd Employee_Attrition_using_SVM

Install the Dependencies:
Ensure that Python is installed on your system (preferably version 3.7 or above). Then, install the necessary Python packages by running:

sh

pip install -r requirements.txt

Prepare the Data:
Make sure your dataset is available and correctly formatted. You may need to modify the data_loader.py script to match your data structure, ensuring it loads and preprocesses the data as required.

Run the Main Script:
Execute the main script to train the SVM model and evaluate its performance:

sh

python employee_attrition_using_svm/main.py

View Results:
The script will output classification metrics such as accuracy, precision, recall, and the confusion matrix. Analyze these results to understand the model's performance and identify areas for improvement.
## Dependencies
The project relies on several Python libraries, which are specified in the requirements.txt file. Key dependencies include:

    pandas: For data manipulation and analysis.
    numpy: For numerical operations and array handling.
    scikit-learn: For implementing the SVM algorithm, data preprocessing, and model evaluation.
    matplotlib: For plotting and visualizing the results.
    seaborn: For creating more detailed visualizations, such as the confusion matrix.

To install these dependencies, run:

sh

pip install -r requirements.txt
