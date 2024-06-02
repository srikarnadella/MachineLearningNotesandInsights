import numpy as np
import pandas as pd
from sklearn import tree
import os
path = r"H:\CodingProjects\MLCourse\Product"
input_file = "PastHires.csv"
source = os.path.join(path,input_file)
df = pd.read_csv(source, header = 0)

print(df.head())

#For the decision tree to work all the data needs to be numerical
#Created a dictionary that maps Y to 1 and N to 0
yndict = {'Y': 1, 'N': 0}

#Goes through the hired column and swaps the vals using our dictionary
df['Hired'] = df['Hired'].map(yndict)
df['Employed?'] = df['Employed?'].map(yndict)
df['Top-tier school'] = df['Top-tier school'].map(yndict)
df['Interned'] = df['Interned'].map(yndict)

educationdict = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(educationdict)

print(df.head())

#Cols: Years experience, Employed, Previous Employers, Levels of Education, Top Tier School, Interned, Hired

#Features: Years experience, Employed, Previous Employers, Levels of Education, Top Tier School, Interned
features = list(df.columns[:6])

#Makes the decision tree
y = df["Hired"]
X = df[features]
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X,y)


#Viewing the decision tree

from IPython.display import Image  
from io import StringIO
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(dtc, out_file=dot_data, feature_names=features)  

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Save the decision tree visualization as a PNG file
graph.write_png("decision_tree.png")

# Open the saved PNG file using the default image viewer
os.system("start decision_tree.png") 


#Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10)
rfc = rfc.fit(X, y)

#Predict employment of an employed 10-year veteran
print (rfc.predict([[10, 1, 4, 0, 0, 0]]))
#...and an unemployed 10-year veteran
print (rfc.predict([[10, 0, 4, 0, 0, 0]]))