import torch
import shap
import matplotlib.pyplot as plt
import os
import pandas as pd

PATH = 'path to file'
model = torch.load(PATH, weights_only=False)


train = pd.read_csv('./train.csv')
train = train.sample(frac=1)
train = train.set_index('PTID')

test = pd.read_csv('./test.csv')
test = test.sample(frac=1)
test = test.set_index('PTID')

explainer = shap.GradientExplainer(model, torch.Tensor(test.values))
shap_values = explainer.shap_values(torch.Tensor(test.values))
print(shap_values[:, :, 0].shape, torch.Tensor(test.values).shape)

shap.summary_plot(shap_values[:, :, 0], features=test.columns, plot_type="bar")

os.environ['KMP_DUPLICATE_LIB_OK']='True'
shap_values_selected = shap_values[:, :, 0]
mean_abs_shap_values = np.mean(np.abs(shap_values_selected), axis=0)
# feature_names = data.feature_names
sorted_idx = np.argsort(mean_abs_shap_values)[::-1]

# Sort the feature names and SHAP values
sorted_feature_names = test.columns[sorted_idx]
sorted_shap_values = mean_abs_shap_values[sorted_idx]

# Create a bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(sorted_feature_names)), sorted_shap_values)
plt.xticks(range(len(sorted_feature_names)), sorted_feature_names, rotation=90)
plt.ylabel('Mean Absolute SHAP Value')
plt.title('Feature Importance based on SHAP Values')

# Add SHAP values on top of the bars
for bar, val in zip(bars, sorted_shap_values):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/8, yval, round(val, 3), va='bottom')  # va: vertical alignment

plt.show()

shap.initjs()  # 初始化JS
shap.force_plot(explainer.expected_value[0], shap_values[:, :, 0], test, show=False)
shap.decision_plot(explainer.expected_value[0], shap_values[:, :, 0], test, show=False)
