# 6_skills_for_data_scientists 
Professionals in data science are expected to excel in six crucial skills: Python programming, global distribution of packages via PyPI, comprehension of generative AI to aid human tasks, application of diverse metrics like confusion matrices, Chi-squared tests with p-values and Spearman's correlation with p-values or Kendall's tau with p-values, execution of advanced internet searches using operators, and the creation of datasets.

When analyzing data, we must determine whether ground truth values are available. If ground truth values are absent, careful consideration is required. Depending on the data properties, choosing linear or nonlinear as well as parametric or nonparametric methods is crucial for accurate analysis.

Fundamental principles dictate that when analyzing the relationship between the target and features, several key elements are essential. These include understanding the data distribution, examining the statistical relationship between the variables with checking multicollinearity, and assessing the statistical validity through the p-value. Therefore, it is important to employ non-linear and nonparametric methods, along with calculating the p-value.

Different machine learning models employ distinct methodologies for calculating feature importance(s) and this can lead to varying degrees of bias. Feature importances from machine learning models are always inherently biased which is not a non-negligible issue. While machine learning target predictions can be validated against known ground truth values to assess accuracy, feature importances derived from models lack such definitive ground truth references for validation.

Multicollinearity occurs when two or more independent variables in a regression model are highly correlated, which can lead to unreliable coefficient estimates. Multicollinearity Assessment: Correlation Matrix:Spearman's correlation, Variance Inflation Factor (VIF): VIF>10 with threshold=5, Tolerance: <0.1, Condition Index: >30, Principal Component Analysis (PCA).

Data science is a cross-cutting technology that can be adapted and applied to any field.

# Fundamental principles on machine learning
1. The primary objective of machine learning is to accurately predict target outcomes with ground truth values.
2. Feature importance measures in machine learning are inherently biased without ground truth values, as different models utilize distinct methodologies for calculation, resulting in varying degrees of non-negligible bias.
3. Cross-validation using metrics such as AUC, RMSE, and R-squared effectively evaluates target prediction accuracy but does not assess the accuracy of feature importance due to absence of ground truth values.
4. Currently, there is no tool available to measure the accuracy of feature importance due to the absence of ground truth data.
5. SHAP can inherit and amplify existing biases from model when using the explain=SHAP(model) function.
6. To determine true associations or genuine relationships between variables, it is essential to analyze the data distribution, examine the statistical relationships between variables, and evaluate statistical validity through p-values.
7. Choosing between linear and nonlinear, as well as parametric and nonparametric approaches, is crucial for minimizing biases.
8. Spearman's correlation, Kendall's tau, Goodman-Kruskal's gamma, Somers’ D, and Hoeffding's D along with their associated p-values, are essential for accurately calculating true associations between variables.

# Collinearity and interactions
Generalized Variance Inflation Factor (GVIF) Calculation from Correlation Matrix: y=f(x1,x2,...,xn)
The VIF can be computed directly using the correlation matrix derived from the features in your dataset. The simplest way to compute it is by using the reciprocal of (1 - R^2) where (R^2) is obtained from the correlation matrix.
<pre>
Interpretation:

A GVIF of 1 indicates no correlation between the ( j )-th variable and the other variables.
A GVIF between 1 and 5 suggests moderate correlation that may not be problematic.
A GVIF above 5 (or sometimes 10) indicates high collinearity; the variable may need to be removed or combined with other variables.
</pre>

<pre>
<b>1.Python PROGRAMMING</b>
<b>1.1.Python programming:</b>
<a href='https://github.com/y-takefuji/python-novice'>https://github.com/y-takefuji/python-novice</a>
  
<b>1.2.IoT programming:</b>
<a  href='https://github.com/y-takefuji/IoT'>https://github.com/y-takefuji/IoT</a>
  
<b>1.3.Fusing AI and Iot programming:</b>
<a href='https://github.com/y-takefuji/mediapipe_pose'>MediaPipe Pose</a>
<a href='https://github.com/y-takefuji/mediapipe_hand'>MediaPipe hand</a>
<a href='https://github.com/y-takefuji/air_calculator'>aircalc</a>
<a href='https://github.com/y-takefuji/airpiano'>airpiano</a>

<b>2.PyPI applications and their reproducibility:</b>
<a href='https://pypi.org/user/takefuji/'>PyPI examples: https://pypi.org/user/takefuji/</a>
<a href='https://doi.org/10.1016/j.chemolab.2023.104941'>PyPI full tutorial:https://github.com/y-takefuji/agci</a>
<a href='https://doi.org/10.1016/j.napere.2024.100089'>PyPI full tutorial</a>
<a href='https://doi.org/10.3390/ijtm1030019'>PyPI old tutorial</a>
<a href='https://doi.org/10.3390/ijtm2020015'>Set Operation: PyPI old tutorial</a>
<a href='https://doi.org/10.1007/s13721-022-00359-1'>deathdaily</a>
<a href='https://www.softwareimpacts.com/article/S2665-9638(22)00137-3/fulltext'>scorecovid</a>
<a href='https://www.softwareimpacts.com/article/S2665-9638(23)00003-9/fulltext'>hiscovid</a>
<a href='https://doi.org/10.1007/s13721-023-00430-5'>covidlag</a>
<a href='https://doi.org/10.1109/TCSS.2022.3227926'>usscore&jpscore</a>
<a href='https://doi.org/10.1016/j.ahr.2023.100167'>midlife excessive mortality</a>
<a href='https://doi.org/10.1016/j.intimp.2023.109823'>vaccine effect</a>
<a href='https://doi.org/10.1016/j.drup.2023.101039'>bivalent vaccine effect</a>
<a href='https://doi.org/10.1007/s11239-023-02930-7'>phope</a>
<a href='https://doi.org/10.1016/j.drup.2024.101174'>pasero</a>
  
<b>3.How to use Generative AI:</b>
<a href='https://github.com/y-takefuji/generativeAI'>https://github.com/y-takefuji/generativeAI</a>

<b>4.Metrics(confusion matrix), AI fairness, chi-squared with p-value, 
  Spearman's correlation or Kendall's tau with p-value analysis:</b>
<a href='https://doi.org/10.1016/j.jemep.2023.100938'>breast cancer screening</a>
<a href='https://doi.org/10.1016/j.jemep.2024.101025'>lung cancer screening</a>
<a href='https://doi.org/10.1016/j.intimp.2024.112032'>arthritis prevalence</a>
<a href='https://doi.org/10.1016/j.aggp.2024.100025'>diabetes prevalence in Japan</a>
<a href='https://doi.org/10.1016/j.ajp.2023.103736'>mental health by sexual orientation</a>
<a href='https://doi.org/10.1007/s41693-024-00134-w'>AI fairness in shaft excavator</a>
<a href='https://doi.org/10.1016/j.cities.2024.105398'>court case disparity with p-value</a>
<a href='https://doi.org/10.1016/j.jad.2024.10.019'>biased feature importance on suicide</a>
<a href='https://doi.org/10.1016/j.annonc.2024.10.013'>biased feature importance of chemotherapy on oncology</a>
<a href='https://doi.org/10.1016/j.oraloncology.2024.107090'>biased radiomic features on oral oncology</a>
<a href='https://doi.org/10.1016/j.jechem.2024.10.032'>biased features on ternary organic solar cells</a>
<a href='https://doi.org/10.1016/j.atherosclerosis.2024.119049'>biased feature importance on protein research</a>
<a href='https://doi.org/10.1016/j.jinf.2024.106357'>biased feature importance on infection research</a>
<a href='https://doi.org/10.1016/j.clnu.2024.11.031'>biased feature importance on clinical nutrition</a>
<a href='https://doi.org/10.1016/j.bbi.2024.11.036'>biased feature importance on brain behavior</a>
<a href='https://doi.org/10.1016/j.ejim.2024.11.022'>biased feature importance on internal medicine</a>
<a href='https://doi.org/10.1016/j.jclinepi.2024.111619'>biased feature importance on clinical epidemiology</a>
<a href='https://doi.org/10.1016/j.tifs.2024.104853'>biased feature importance on food science</a>
<a href='https://doi.org/10.1016/j.ajog.2024.12.010'>biased feature importance on gynecology</a>
<a href='https://doi.org/10.1016/j.molp.2024.12.014'>biased feature importance on molecular plant</a>
<a href='https://doi.org/10.1016/j.bja.2024.11.033'>biased feature importance on anaesthesia</a>
<a href='https://doi.org/10.1200/PO-24-00785'>biased feature importance on precision oncology</a>
<a href='https://doi.org/10.1016/j.ajem.2025.01.009'>biased feature importance on energency medicine</a>
<a href='https://doi.org/10.1016/j.retram.2024.103490'>data misinterpretation</a>
<a href='https://doi.org/10.1016/j.heha.2024.100109'>contradictions in global co2 on global warming with p-values</a>
<a href='https://doi.org/10.1016/j.cie.2024.110667'>reducing instance bias for AI tacograph</a>

<b>5.How to use internet search engines</b>
https://github.com/y-takefuji/bash-shell
  download keyword.crypted and decrypt it:
  How to decrypt keyword.crypted file.
  For Windows:
$ openssl enc -d -aes256 -pbkdf2 -in keyword.crypted >keyword.pptx
  For Mac:
$ openssl enc -d -aes256 -pbkdf2 -in keyword.crypted >keyword.pptx -md sha256

<a href='https://doi.org/10.1016/j.aggp.2024.100025'>search operator:https://doi.org/10.1016/j.aggp.2024.100025</a>
<a href='https://doi.org/10.3390/ijtm2020015'>set operation:https://doi.org/10.3390/ijtm2020015</a>
<a href='https://doi.org/10.1007/s42824-024-00146-5'>Google Trends</a>

<b>6.How to create datasets for machine learning</b>
<a href='https://doi.org/10.1016/j.aei.2021.101354'>reducing variables</a>
<a href='https://doi.org/10.1007/s41693-024-00134-w'>construction robots</a>

Machine learning is equivalent to forming the relationship function f(): y=f(X) or Y=f(X) 
True associations or genuine relationships between the target and features: 
relationships beween x1 and y, x2 and y,..., xn and y.
3 key elements: the data distribution, examining the statistical relationship between the variables, 
  and assessing the statistical validity through the p-value
  
X: independent variables; X=(x1,x2,...,xn)
y: dependent variable 
Y: dependent variables; Y=(y1,y2,...,ym)
</pre>
