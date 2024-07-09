{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jW_2Ti6D8s7W",
        "outputId": "5578795f-2849-43d6-b724-6d4d29e53565"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-8-a7a8b9b48b8d>:19: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_train['merge_answers'] = merge_answers_train\n",
            "<ipython-input-8-a7a8b9b48b8d>:20: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_train['merge_category'] = merge_category_train\n",
            "<ipython-input-8-a7a8b9b48b8d>:23: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_train.dropna(subset=['/questionText', 'merge_answers', 'merge_category'], inplace=True)\n",
            "<ipython-input-8-a7a8b9b48b8d>:36: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_test['merge_answers'] = merge_answers_test\n",
            "<ipython-input-8-a7a8b9b48b8d>:37: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_test['merge_category'] = merge_category_test\n",
            "<ipython-input-8-a7a8b9b48b8d>:40: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_test.dropna(subset=['/questionText', 'merge_answers', 'merge_category'], inplace=True)\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.model_selection import StratifiedKFold\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report\n",
        "import time\n",
        "\n",
        "# Load the training CSV file into a pandas DataFrame\n",
        "train_data = pd.read_csv(\"/content/drive/MyDrive/Dissertation_Rivesh_Boodhooa_2110917/bettle_model/bettle/test_unseen_ans/Beetle_core_5_way_train.csv\")\n",
        "\n",
        "# Ensure to select the relevant columns for both training and testing datasets\n",
        "df_train = train_data[['/questionText', '/referenceAnswers/referenceAnswer', '/referenceAnswers/referenceAnswer/@category', '/studentAnswers/studentAnswer/@accuracy', '/studentAnswers/studentAnswer']]\n",
        "\n",
        "# Combine columns to create features\n",
        "merge_answers_train = df_train['/studentAnswers/studentAnswer'].combine_first(df_train['/referenceAnswers/referenceAnswer'])\n",
        "merge_category_train = df_train['/studentAnswers/studentAnswer/@accuracy'].combine_first(df_train['/referenceAnswers/referenceAnswer/@category'])\n",
        "\n",
        "# Add combined features to DataFrame\n",
        "df_train['merge_answers'] = merge_answers_train\n",
        "df_train['merge_category'] = merge_category_train\n",
        "\n",
        "# Drop rows with NaN values\n",
        "df_train.dropna(subset=['/questionText', 'merge_answers', 'merge_category'], inplace=True)\n",
        "\n",
        "# Load the testing CSV file into a pandas DataFrame\n",
        "test_data = pd.read_csv(\"/content/drive/MyDrive/Dissertation_Rivesh_Boodhooa_2110917/bettle_model/bettle/test_unseen_ans/test_unseen_question.csv\")\n",
        "\n",
        "# Ensure to select the relevant columns for testing dataset\n",
        "df_test = test_data[['/questionText', '/referenceAnswers/referenceAnswer', '/referenceAnswers/referenceAnswer/@category', '/studentAnswers/studentAnswer/@accuracy', '/studentAnswers/studentAnswer']]\n",
        "\n",
        "# Combine columns to create features\n",
        "merge_answers_test = df_test['/studentAnswers/studentAnswer'].combine_first(df_test['/referenceAnswers/referenceAnswer'])\n",
        "merge_category_test = df_test['/studentAnswers/studentAnswer/@accuracy'].combine_first(df_test['/referenceAnswers/referenceAnswer/@category'])\n",
        "\n",
        "# Add combined features to DataFrame\n",
        "df_test['merge_answers'] = merge_answers_test\n",
        "df_test['merge_category'] = merge_category_test\n",
        "\n",
        "# Drop rows with NaN values\n",
        "df_test.dropna(subset=['/questionText', 'merge_answers', 'merge_category'], inplace=True)\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# Define the paths to save the new datasets\n",
        "#train_save_path = '/content/drive/MyDrive/Dissertation_Rivesh_Boodhooa_2110917/bettle_model/bettle/test_unseen_ans/traindataset.csv'\n",
        "#test_save_path = '/content/drive/MyDrive/Dissertation_Rivesh_Boodhooa_2110917/bettle_model/bettle/test_unseen_ans/testdataset.csv'\n",
        "# Select the relevant columns to save\n",
        "#df_train_filtered = df_train[['/questionText', 'merge_answers', 'merge_category']]\n",
        "#df_test_filtered = df_test[['/questionText', 'merge_answers', 'merge_category']]\n",
        "# Save the new training dataset with selected columns\n",
        "#df_train_filtered.to_csv(train_save_path, index=False)\n",
        "# Save the new testing dataset with selected columns\n",
        "#df_test_filtered.to_csv(test_save_path, index=False)\n",
        "#print(\"Filtered datasets saved successfully.\")\n"
      ],
      "metadata": {
        "id": "T78g3ngqQs9j"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# Select the relevant columns for training\n",
        "df_train_filtered = df_train[['/questionText', 'merge_answers', 'merge_category']]\n",
        "df_test_filtered = df_test[['/questionText', 'merge_answers', 'merge_category']]\n",
        "\n",
        "# Combine question text and answers into a single text feature\n",
        "df_train_filtered['text'] = df_train_filtered['merge_answers'] + ' ' + df_train_filtered['merge_category']\n",
        "df_test_filtered['text'] = df_test_filtered['merge_answers'] + ' ' + df_test_filtered['merge_category']\n",
        "\n",
        "# Define the features and target\n",
        "X_train = df_train_filtered['text']\n",
        "y_train = df_train_filtered['/questionText']\n",
        "X_test = df_test_filtered['text']\n",
        "y_test = df_test_filtered['/questionText']\n",
        "\n",
        "# Vectorization\n",
        "tfidf = TfidfVectorizer()\n",
        "X_train_tfidf = tfidf.fit_transform(X_train)\n",
        "X_test_tfidf = tfidf.transform(X_test)\n",
        "\n",
        "# Stratified K-Fold setup\n",
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "model = LogisticRegression(random_state=42, max_iter=10000)\n",
        "\n",
        "# Track training time\n",
        "start_time = time.time()\n",
        "\n",
        "# Train the model using StratifiedKFold to handle imbalance\n",
        "for train_index, val_index in skf.split(X_train_tfidf, y_train):\n",
        "    X_train_fold, X_val_fold = X_train_tfidf[train_index], X_train_tfidf[val_index]\n",
        "    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]\n",
        "    model.fit(X_train_fold, y_train_fold)\n",
        "\n",
        "# End tracking training time\n",
        "end_time = time.time()\n",
        "train_time = end_time - start_time\n",
        "\n",
        "# Model evaluation\n",
        "start_eval_time = time.time()\n",
        "predictions_train = model.predict(X_train_tfidf)\n",
        "predictions_test = model.predict(X_test_tfidf)\n",
        "end_eval_time = time.time()\n",
        "eval_time = end_eval_time - start_eval_time\n",
        "\n",
        "accuracy_train = accuracy_score(y_train, predictions_train)\n",
        "accuracy_test = accuracy_score(y_test, predictions_test)\n",
        "precision_test = precision_score(y_test, predictions_test, average='weighted', zero_division=1)\n",
        "f1_test = f1_score(y_test, predictions_test, average='weighted')\n",
        "recall_test = recall_score(y_test, predictions_test, average='weighted', zero_division=1)\n",
        "\n",
        "# Output metrics\n",
        "print(\"####################################################### First Dataset ###################################################################\")\n",
        "print(\"Training Accuracy:\", accuracy_train)\n",
        "print(\"Test Accuracy:\", accuracy_test)\n",
        "print(\"Test Precision:\", precision_test)\n",
        "print(\"Test F1 Score:\", f1_test)\n",
        "print(\"Test Recall:\", recall_test)\n",
        "print(\"Time taken to train:\", train_time, \"seconds\")\n",
        "print(\"Time taken to evaluate:\", eval_time, \"seconds\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sFSAmTGLQs1L",
        "outputId": "b578ad67-af2e-45b7-8f7b-5ea23a57a981"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-9-b1cb8fc3ccb2>:6: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_train_filtered['text'] = df_train_filtered['merge_answers'] + ' ' + df_train_filtered['merge_category']\n",
            "<ipython-input-9-b1cb8fc3ccb2>:7: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  df_test_filtered['text'] = df_test_filtered['merge_answers'] + ' ' + df_test_filtered['merge_category']\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "####################################################### First Dataset ###################################################################\n",
            "Training Accuracy: 0.5746471898984897\n",
            "Test Accuracy: 0.17093023255813952\n",
            "Test Precision: 0.8233554817275748\n",
            "Test F1 Score: 0.10845949742469173\n",
            "Test Recall: 0.17093023255813952\n",
            "Time taken to train: 7.396774053573608 seconds\n",
            "Time taken to evaluate: 0.0158998966217041 seconds\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "i504PysNQsiD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "final\n"
      ],
      "metadata": {
        "id": "4Kfv_mtMMl6M"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score\n",
        "from sklearn.model_selection import StratifiedKFold\n",
        "import time\n",
        "\n",
        "# Load the training CSV file into a pandas DataFrame\n",
        "df_train = pd.read_csv(\"/content/drive/MyDrive/Dissertation_Rivesh_Boodhooa_2110917/bettle_model/bettle/test_unseen_ans/Beetle_core_5_way_train.csv\")\n",
        "\n",
        "# Drop rows with any missing values in the specified columns\n",
        "df_train.dropna(subset=['/questionText', '/studentAnswers/studentAnswer/@accuracy', '/studentAnswers/studentAnswer'], inplace=True)\n",
        "\n",
        "# Combine text features\n",
        "X_train = df_train['/questionText'] + ' ' + df_train['/studentAnswers/studentAnswer']\n",
        "y_train = df_train['/studentAnswers/studentAnswer/@accuracy']\n",
        "\n",
        "# Load the testing CSV file into a pandas DataFrame\n",
        "df_test = pd.read_csv(\"/content/drive/MyDrive/Dissertation_Rivesh_Boodhooa_2110917/bettle_model/bettle/test_unseen_ans/test_unseen_question.csv\")\n",
        "\n",
        "# Drop rows with any missing values in the specified columns\n",
        "df_test.dropna(subset=['/questionText', '/studentAnswers/studentAnswer/@accuracy', '/studentAnswers/studentAnswer'], inplace=True)\n",
        "\n",
        "# Combine text features\n",
        "X_test = df_test['/questionText'] + ' ' + df_test['/studentAnswers/studentAnswer']\n",
        "y_test = df_test['/studentAnswers/studentAnswer/@accuracy']\n",
        "\n",
        "# Vectorization\n",
        "tfidf = TfidfVectorizer()\n",
        "X_train_tfidf = tfidf.fit_transform(X_train)\n",
        "X_test_tfidf = tfidf.transform(X_test)\n",
        "\n",
        "# Stratified K-Fold setup\n",
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "model = RandomForestClassifier(random_state=42, n_estimators=100)\n",
        "\n",
        "# Track training time\n",
        "start_time = time.time()\n",
        "\n",
        "# Train the model using StratifiedKFold to handle imbalance\n",
        "for train_index, val_index in skf.split(X_train_tfidf, y_train):\n",
        "    X_train_fold, X_val_fold = X_train_tfidf[train_index], X_train_tfidf[val_index]\n",
        "    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]\n",
        "    model.fit(X_train_fold, y_train_fold)\n",
        "\n",
        "# Train the final model on the entire training dataset\n",
        "model.fit(X_train_tfidf, y_train)\n",
        "\n",
        "# End tracking training time\n",
        "end_time = time.time()\n",
        "train_time = end_time - start_time\n",
        "\n",
        "# Model evaluation\n",
        "start_eval_time = time.time()\n",
        "predictions_train = model.predict(X_train_tfidf)\n",
        "predictions_test = model.predict(X_test_tfidf)\n",
        "end_eval_time = time.time()\n",
        "eval_time = end_eval_time - start_eval_time\n",
        "\n",
        "accuracy_train = accuracy_score(y_train, predictions_train)\n",
        "accuracy_test = accuracy_score(y_test, predictions_test)\n",
        "precision_test = precision_score(y_test, predictions_test, average='weighted', zero_division=1)\n",
        "f1_test = f1_score(y_test, predictions_test, average='weighted')\n",
        "recall_test = recall_score(y_test, predictions_test, average='weighted', zero_division=1)\n",
        "\n",
        "# Output metrics\n",
        "print(\"####################################################### First Dataset ###################################################################\")\n",
        "print(\"Training Accuracy:\", accuracy_train)\n",
        "print(\"Test Accuracy:\", accuracy_test)\n",
        "print(\"Test Precision:\", precision_test)\n",
        "print(\"Test F1 Score:\", f1_test)\n",
        "print(\"Test Recall:\", recall_test)\n",
        "print(\"Time taken to train:\", train_time, \"seconds\")\n",
        "print(\"Time taken to evaluate:\", eval_time, \"seconds\")\n",
        "print(\"##########################################################################################################################################\")\n",
        "print(\"\\n\")\n",
        "\n",
        "print(\"####################################################### Second Dataset ###################################################################\")\n",
        "# Load the second testing CSV file into a pandas DataFrame\n",
        "df_test2 = pd.read_csv(\"/content/drive/MyDrive/Dissertation_Rivesh_Boodhooa_2110917/bettle_model/bettle/test_unseen_ans/test_unseen_answer.csv\", skiprows=[0])\n",
        "\n",
        "# Handle missing values\n",
        "df_test2.dropna(subset=['/questionText', '/studentAnswers/studentAnswer/@accuracy', '/studentAnswers/studentAnswer'], inplace=True)\n",
        "\n",
        "# Combine text features\n",
        "X_test2 = df_test2['/questionText'] + ' ' + df_test2['/studentAnswers/studentAnswer']\n",
        "y_test2 = df_test2['/studentAnswers/studentAnswer/@accuracy']\n",
        "\n",
        "# Vectorization\n",
        "X_test_tfidf2 = tfidf.transform(X_test2)\n",
        "\n",
        "# Train the final model on the entire second dataset\n",
        "model2 = RandomForestClassifier(random_state=42, n_estimators=100)\n",
        "start_time2 = time.time()\n",
        "model2.fit(X_test_tfidf2, y_test2)\n",
        "end_time2 = time.time()\n",
        "train_time2 = end_time2 - start_time2\n",
        "\n",
        "# Model evaluation for second dataset\n",
        "start_eval_time2 = time.time()\n",
        "predictions_test2 = model2.predict(X_test_tfidf2)\n",
        "end_eval_time2 = time.time()\n",
        "eval_time2 = end_eval_time2 - start_eval_time2\n",
        "\n",
        "accuracy_test2 = accuracy_score(y_test2, predictions_test2)\n",
        "precision_test2 = precision_score(y_test2, predictions_test2, average='weighted', zero_division=1)\n",
        "f1_test2 = f1_score(y_test2, predictions_test2, average='weighted')\n",
        "recall_test2 = recall_score(y_test2, predictions_test2, average='weighted', zero_division=1)\n",
        "\n",
        "# Output metrics for second dataset\n",
        "print(\"Training Accuracy (second dataset):\", accuracy_train)\n",
        "print(\"Test Accuracy (second dataset):\", accuracy_test2)\n",
        "print(\"Test Precision (second dataset):\", precision_test2)\n",
        "print(\"Test F1 Score (second dataset):\", f1_test2)\n",
        "print(\"Test Recall (second dataset):\", recall_test2)\n",
        "print(\"Time taken to train (second dataset):\", train_time2, \"seconds\")\n",
        "print(\"Time taken to evaluate (second dataset):\", eval_time2, \"seconds\")\n",
        "print(\"##########################################################################################################################################\")\n"
      ],
      "metadata": {
        "id": "NM8bye8M5lsr",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "37582a96-72c7-4102-f0c8-c9f7ecbafe3e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "####################################################### First Dataset ###################################################################\n",
            "Training Accuracy: 0.9783797864027091\n",
            "Test Accuracy: 0.5116279069767442\n",
            "Test Precision: 0.5200858217469744\n",
            "Test F1 Score: 0.5013916805465422\n",
            "Test Recall: 0.5116279069767442\n",
            "Time taken to train: 36.97487807273865 seconds\n",
            "Time taken to evaluate: 0.16939973831176758 seconds\n",
            "##########################################################################################################################################\n",
            "\n",
            "\n",
            "####################################################### Second Dataset ###################################################################\n",
            "Training Accuracy (second dataset): 0.9783797864027091\n",
            "Test Accuracy (second dataset): 0.9932279909706546\n",
            "Test Precision (second dataset): 0.9933390075121192\n",
            "Test F1 Score (second dataset): 0.9932095908555517\n",
            "Test Recall (second dataset): 0.9932279909706546\n",
            "Time taken to train (second dataset): 0.581660270690918 seconds\n",
            "Time taken to evaluate (second dataset): 0.02675795555114746 seconds\n",
            "##########################################################################################################################################\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QoK2PdSV8oXE",
        "outputId": "45938b7a-93dc-4b4f-d034-0eaf15a5d6ff"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1MbBTEJM32P3iDCptZaRyAMg4vUb2Lj6B",
      "authorship_tag": "ABX9TyPV2UUfMQ3p3HNvMSL7g5DB"
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}