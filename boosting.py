import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    
   dt_heart = pd.read_csv('./data/heart.csv')

   print(dt_heart['target'].describe())

   X = dt_heart.drop(['target'], axis=1)
   y = dt_heart['target']

   X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35)

   boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
   boost_pred = boost.predict(X_test)
   print("-"*64)
   print(accuracy_score(boost_pred, y_test))
   estimators = range(10, 200, 10)
   total_accuracy = {}
   for i in estimators:
        boost = GradientBoostingClassifier(n_estimators=i).fit(X_train, y_train)
        boost_pred = boost.predict(X_test)

        total_accuracy[i] = accuracy_score(y_test, boost_pred)
    
   plt.plot(estimators, total_accuracy.values())
   plt.xlabel('Estimators')
   plt.ylabel('Accuracy')
   plt.show()
   plt.savefig('Boost.png')

   max_accu = np.array(list(total_accuracy.values())).max()
   best_est_num = {k: v for k, v in total_accuracy.items() if v == max_accu}
   best_est_num = list(best_est_num)
   print(best_est_num[0])
#implementacion_boosting