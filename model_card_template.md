# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised machine learning classifier trained to predict whether an individual earns more than $50,000 per year using the UCI Adult Census dataset. The model is implemented as a Random Forest Classifier and uses one‑hot encoding for categorical features and label binarization for the target variable. The preprocessing artifacts (encoder and label binarizer) and the trained model are saved for reproducibility and deployment.

## Intended Use
The model is intended for educational purposes as part of demonstrating an end‑to‑end MLOps workflow, including data processing, model training, evaluation, CI/CD, and API deployment. It is not intended for real‑world decision‑making in hiring, credit scoring, or financial assessment. Any real‑world use would require extensive fairness testing, bias mitigation, and regulatory review.

## Training Data
The model was trained on the Adult Census dataset, which includes demographic and employment‑related features such as age, workclass, education, marital status, occupation, relationship, race, sex, and native country. The dataset was split into training and testing sets using an 80/20 split. Categorical features were processed using one‑hot encoding, and the target variable was binarized.

## Evaluation Data
The evaluation data consists of the held‑out test split from the same dataset. The same preprocessing pipeline used during training was applied to the evaluation data. Model performance was assessed using standard classification metrics.

## Metrics
The model was evaluated using precision, recall, and F1 score:

Precision: 0.7419

Recall: 0.6384

F1 Score: 0.6863

Slice‑based performance was also computed across categorical features. Performance varies across groups, with some slices (e.g., “Doctorate”, “Prof‑school”, “Exec‑managerial”) showing strong performance, while others with small sample sizes or imbalanced labels (e.g., “7th‑8th grade”, “Greece”, “Jamaica”) show poor or unstable metrics.

## Ethical Considerations
The dataset includes sensitive attributes such as race, sex, and marital status. These features can introduce bias into the model’s predictions. Some demographic groups have significantly fewer samples, leading to unreliable slice metrics and potential fairness concerns. The model should not be used for decisions that affect individuals’ opportunities or livelihoods without thorough fairness analysis and mitigation strategies.

## Caveats and Recommendations
The dataset is from the 1990s and may not reflect current socioeconomic patterns. Some categories contain very few samples, resulting in unstable slice metrics. The model does not account for temporal changes or external socioeconomic factors. For any real‑world use, additional data validation, fairness testing, and model monitoring would be required.
