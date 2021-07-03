# attrition-heroku

Hiring & retraining employees require substantial time, skills and capital and this can impact small businesses particularly. High employee attrition will lead to significant disruption and utimately higher costs. Therefore it is in the interests of a business to understand the drivers of attrition and take steps to minimise employee attrition.

Using models to predict if a staff member is likely to quit can support HR's ability to be proactive and to take actions to prevent or at least minimise the probability of the employee leaving.

This data set was obtained from a IBM survey and is available here https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset¶
The data will of course be relevant to the company that gathered the data, however as a future development, companies and businesses can gather data more relevent to their line of business hence providing a platform for generating new insights and what drives attrition in their business.

The following Models were trained and evaluated:
Logistic Regression
Random Forest
Deep Learning (with tensorflow)
XGB Classifier
Random Forest, Deep Learning and XGB Classifier had siimilar accuracy scores (with SMOTE applied). The Deep Learniing model takes a little longer to process and uses up more memory so this model was excluded. In the end RF and XGB had similar performance so either of them would have been suffice. In this case I choose XGB and this will be the model deployed.

Note, the dataset used to build this application was unbalanced. When training models on such datasets, class unbalance influences a learning algorithm during training by making decision rule biased towards the majority class and optimizes the predictions based on the majority class in the dataset. There are are a number of ways to deal with this issue and in this case SMOTE (Synthetic Minority Over-sampling Technique) was used. This method creates synthetic samples of your data, so rather than taking copies of observations, SMOTE uses a distance measure to create synthetic samples of data points that would not be far from the data points. Refer to the Feature engineering and model evaluation Jupyter Notebook for more details.
