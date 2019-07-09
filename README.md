# Project:User Knowledge Modelling

- ***Estimation of students' knowledge status about the subject of Electrical DC Machines.***


**Abstract:**

Creating an efficient user knowledge model is a crucial task for web-based adaptive learning environments in different domains. It is often a challenge to determine exactly what type of domain dependent data will be stored and how it will be evaluated by a user modelling system. The most important disadvantage of these models is that they classify the knowledge of users without taking into account the weight differences among the domain dependent data of users. For this purpose, both the probabilistic and the instance-based models have been developed and commonly used in the user modelling systems. In this study a powerful, efficient and simple ‘Intuitive Knowledge Classifier’ method is proposed and presented to model the domain dependent data of users. A domain independent object model, the user modelling approach and the weight-tuning method are combined with instance-based classification algorithm to improve classification performances of well-known the Bayes and the k-nearest neighbour-based methods. The proposed knowledge classifier intuitively explores the optimum weight values of students’ features on their knowledge class first. Then it measures the distances among the students depending on their data and the values of weights. Finally, it uses the dissimilarities in the classification process to determine their knowledge class. The experimental studies have shown that the weighting of domain dependent data of students and combination of user modelling algorithms and population-based searching approach play an essential role in classifying performance of user modelling system. The proposed system improves the classification accuracy of instance-based user modelling approach for all distance metrics and different k-values.

**Attribute Information:**

- STG (The degree of study time for goal object materails), (input value) 
- SCG (The degree of repetition number of user for goal object materails) (input value) 
- STR (The degree of study time of user for related objects with goal object) (input value) 
- LPR (The exam performance of user for related objects with goal object) (input value) 
- PEG (The exam performance of user for goal objects) (input value) 
- UNS (The knowledge level of user) (target value) 
  - Very Low: 50 
  - Low:129 
  - Middle: 122 
  - High 130
