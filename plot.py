import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def comparative_plot(confidence_SVM,
                     confidence_KNN,
                     confidence_DTC,
                     confidence_MLP,
                     confidence_CNN,
                     confidence_Hybrid,
                     confidence_Hybrid2):
    # model_columns is an empty list that will be used to create the DataFrame
    model_columns = []
    # model_compare is an empty Pandas DataFrame with no rows or columns
    model_compare = pd.DataFrame(columns=model_columns)
    # model_Name is a list of the names of the algorithms being compared
    model_Name = [
        'SVM',
        'KNN',
        'DTC',
        'MLP',
        'CNN',
        'Hybrid',
        'Hybrid2'
    ]
    # accuracies is a list of the accuracies for each of the algorithms
    accuracies = [
        confidence_SVM,
        confidence_KNN,
        confidence_DTC,
        confidence_MLP,
        confidence_CNN,
        confidence_Hybrid,
        confidence_Hybrid2
    ]
    # Initialize counters for the loop
    count = 0
    row_index = 0
    # Iterate over the algorithm names and accuracies
    for i in model_Name:
        # model_name is the name of the current algorithm
        model_name = model_Name[count]
        # Add the algorithm name to the 'Name' column of the DataFrame
        model_compare.loc[row_index, 'Name'] = model_name
        # Add the accuracy to the 'Accuracies' column of the DataFrame
        model_compare.loc[row_index, 'Accuracies'] = accuracies[count]

        # Increment the counters
        count += 1
        row_index += 1

    # Sort the DataFrame by the 'Accuracies' column in descending order
    model_compare.sort_values(by=['Accuracies'], ascending=False, inplace=True)

    # Save the DataFrame as a CSV file
    model_compare.to_csv('static/assets/csv/compare.csv', encoding='utf-8-sig')

    # Create a bar plot of the algorithm accuracies using Seaborn
    plt.subplots(figsize=(15, 6))
    sns.barplot(x="Name", y="Accuracies", data=model_compare, palette='hot', edgecolor=sns.color_palette('dark', 7))
    plt.xticks(rotation=90)
    plt.title('Accuracies Comparison')
    # Save the plot as an image file
    plt.savefig('static/assets/img/model_compare.jpg')

    # Return the DataFrame
    return model_compare


# # Test
# confidence_SVM = 0.99
# confidence_KNN = 0.99
# confidence_DT = 0.99
# confidence_ANN = 0.67
# confidence_CNN = 0.67
# confidence_Hybrid = 0.67
# confidence_Hybrid2 = 0.67
#
# comparative_plot(confidence_SVM,
#                  confidence_KNN,
#                  confidence_DT,
#                  confidence_ANN,
#                  confidence_CNN,
#                  confidence_Hybrid,
#                  confidence_Hybrid2)
