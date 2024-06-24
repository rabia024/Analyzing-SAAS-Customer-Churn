# Analyzing-SAAS-Customer-Churn
The dilemma we face is the high churn rate observed among the subscriber base. The subscription-
based service industry faces a significant challenge in retaining customers over time. High
customer churn rates not only impact revenue but also indicate potential issues with customer
satisfaction, service quality, or competitive pressures. Identifying customers who are likely to
churn before they actually do so is crucial for implementing targeted retention strategies and
improving overall customer loyalty. I aim to address this challenge by leveraging data
analytics.
Dataset Overview:
•	Columns: There are 21 columns in the data frame. The first column contains row indices. The remaining columns contain different customer data points.
•	Rows: The data frame contains 243,787 rows, which likely represents the number of customers the company has.
•	Data Types: The data types include numeric data (e.g., account age, monthly charges, total charges) and categorical data (e.g., subscription type, payment method, genre preference).
Key insights:
Churn rate by Account Age
It appears that churn is highest for customers in the first month and then levels out over time. This suggests that the first month is critical for retaining new customers.
Churn Rate by Payment Method. Also It appears that customers who pay by credit card have a lower churn rate than those who use other payment methods
 Visualized a list of top priority customer IDs, potentially based on High average views per week. It focuses retention efforts on these high-value customers by providing them with personalized recommendations. By comparing the content download rates across segments, we can identify which customer groups engage more with downloads. This informs decisions about content recommendations or targeted marketing campaigns.

Conclusion:
After conducting an in-depth analysis of customer churn within our subscription service, several key findings have emerged. Firstly, it is evident that customer churn poses a significant challenge for our business, impacting customer retention rates. The churn rate, calculated as the percentage of customers who terminated their subscriptions during a specific period, provides valuable insights into customer behavior. We observed monthly charges, payment method, subscription type had an impact on customer churn. Further more using ML we made a model to predict customer churn data on the trained model and extracted high likely to churn customer Ids. By targeting high-risk customers with personalized retention tactics, such as targeted promotions, loyalty programs, and proactive customer support, we can mitigate churn and improve customer lifetime 


