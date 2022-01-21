#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc

from xgboost import XGBClassifier, plot_importance

import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv("Datos/original.csv")
df = df.drop(["Pressure(in)","Station", "End_Time", "Start_Lat", "Start_Lng", "End_Lat", "End_Lng", "Description", "Number", "Traffic_Calming", "Street", "Side", "Zipcode", "Country", "Timezone", "Airport_Code", "Weather_Timestamp", "Wind_Chill(F)", "Sunrise_Sunset", "Nautical_Twilight", "Astronomical_Twilight"], axis = 1)
df.dropna(inplace=True)
df.head()


# In[6]:


accidentes = df.copy()


# In[7]:


target = 'Severity'
features_list = list(accidentes.columns)
features_list.remove(target)
print(features_list)


# In[8]:


from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()

df['ID'] = lbl.fit_transform(df['ID'].astype(str))
df['Start_Time'] = lbl.fit_transform(df['Start_Time'].astype(str))
df['City'] = lbl.fit_transform(df['City'].astype(str))
df['County'] = lbl.fit_transform(df['County'].astype(str))
df['State'] = lbl.fit_transform(df['State'].astype(str))
df['Wind_Direction'] = lbl.fit_transform(df['Wind_Direction'].astype(str))
df['Weather_Condition'] = lbl.fit_transform(df['Weather_Condition'].astype(str))
df['Civil_Twilight'] = lbl.fit_transform(df['Civil_Twilight'].astype(str))

accidentes[features_list].hist(bins=40, edgecolor='b', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False, figsize=(16,6), color='red')    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))   
plt.suptitle('Accidentes Univariate Plots', x=0.65, y=1.25, fontsize=14);  


# # Distribución de outcomes

# In[9]:


accidentes[target].hist(bins=40, edgecolor='b', linewidth=1.0,
              xlabelsize=8, ylabelsize=8, grid=False, figsize=(6,2), color='red')    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))   
plt.suptitle('Accidentes Plot', x=0.65, y=1.25, fontsize=14);  


# In[10]:


f, ax = plt.subplots(figsize=(10, 6))
corr = accidentes.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="Blues",fmt='.2f',
            linewidths=.05)
f.subplots_adjust(top=0.93)
t = f.suptitle('Accidentes Attributes Correlation Heatmap', fontsize=12)


# # Características bivariadas frente al resultado

# In[8]:


df2 = pd.read_csv("Datos/test3.csv")
df2 = df2.drop(["Pressure(in)", "End_Time", "Start_Lat", "Start_Lng", "End_Lat", "End_Lng", "Description", "Number", "Street", "Side", "Zipcode", "Traffic_Calming", "Country", "Timezone", "Airport_Code", "Weather_Timestamp", "Wind_Chill(F)", "Sunrise_Sunset", "Nautical_Twilight"], axis = 1)
df2.dropna(inplace=True)

df2['Civil_Twilight'] = df2['Civil_Twilight'].astype('category').cat.codes
df2['State'] = df2['State'].astype('category').cat.codes

sns.pairplot( df2.dropna(), vars=[ 'Temperature(F)', 'Humidity(%)', 'State', 'Civil_Twilight' ], size=2, diag_kind='kde', palette='hls', hue='Severity' )
plt.tight_layout( )
plt.show()


# # Entrenamiento modelo

# In[11]:


accidentes = df.copy()


# In[12]:


X = accidentes.iloc[:, df.columns != 'Severity']
y = accidentes.iloc[:, df.columns == 'Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train.shape, X_test.shape

y_train.head()


# In[13]:


get_ipython().run_cell_magic('time', '', " \nxgb = XGBClassifier(objective='binary:logistic', random_state=33)\nxgb.fit(X_train, y_train, eval_metric = 'logloss')")


# # Evaluación del rendimiento del modelo

# In[14]:


xgb_predictions = xgb.predict(X_test)


# In[23]:


def evaluation_scores(test, prediction, target_names=None):
    print('Precisión:', np.round(metrics.accuracy_score(test, prediction), 4)) 
    print('-'*60)
    print('Informe de clasificación:\n\n', metrics.classification_report(y_true=test, y_pred=prediction, target_names=target_names)) 
    
    classes = [0, 1]
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y_true=test, y_pred=prediction, labels=classes)
    print('-'*60)
    print('Matriz de confusión:\n')
    print(cm) 


# In[24]:


evaluation_scores(y_test, xgb_predictions, target_names=['1', '2', '3', '4'])


# # Clasificacion ROC y AUC

# In[17]:


probs = xgb.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'red', label = 'ROC AUC score = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'b--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Métodos de interpretación

# In[18]:


fig = plt.figure(figsize = (18, 10))
title = fig.suptitle("Importancia de las características nativas de XGBoost", fontsize=14)

ax1 = fig.add_subplot(2, 2, 1)
plot_importance(xgb, importance_type='weight', ax=ax1, color='red')
ax1.set_title("Importancia de las características con su peso");

ax2 = fig.add_subplot(2, 2, 2)
plot_importance(xgb, importance_type='cover', ax=ax2, color='red')
ax2.set_title("Importancia de la característica con la cobertura de la muestra");

ax3 = fig.add_subplot(2, 2, 3)
plot_importance(xgb, importance_type='gain', ax=ax3, color='red')
ax3.set_title("Importancia de las características con ganancia media");


# ## ELI5 Model Interpretation

# In[19]:


import eli5
from eli5.sklearn import PermutationImportance


# In[20]:


eli5.show_weights(xgb.get_booster())


# In[21]:


get_ipython().run_cell_magic('time', '', "\nxgb_array = XGBClassifier(objective='binary:logistic', random_state=33, n_jobs=-1)\nxgb_array.fit(X_train, y_train, eval_metric = 'logloss')")


# In[22]:


feat_permut = PermutationImportance(xgb_array, random_state=33).fit(X_train, y_train)
eli5.show_weights(feat_permut, feature_names = features_list)


# ## Partial Dependence Plots (PD plot)

# In[3]:


from pdpbox import pdp, get_dataset, info_plots

def plot_pdp(model, df, feature, cluster_flag=False, nb_clusters=None, lines_flag=False):
    
    pdp_goals = pdp.pdp_isolate(model=model, dataset=df, model_features=df.columns.tolist(), feature=feature)

    # plot it
    pdp.pdp_plot(pdp_goals, feature, cluster=cluster_flag, n_cluster_centers=nb_clusters, plot_lines=lines_flag)
    plt.show()


# In[24]:


plot_pdp(xgb, X_train, 'Humidity(%)')


# ### Gráfico ICE univariante

# In[33]:


plot_pdp(xgb, X_train, 'State', cluster_flag=True, nb_clusters=24, lines_flag=True)


# ### Gráfico PD Bivariante

# In[35]:


get_ipython().run_cell_magic('time', '', "\nfeatures_to_plot = ['State', 'Humidity(%)']\ninter1  =  pdp.pdp_interact(model=xgb, dataset=X_train, model_features=features_list, features=features_to_plot)\npdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='grid')\n\nplt.show()")


# ## SKATER Model Interpretation

# In[25]:


from skater.core.explanations import Interpretation
from skater.model import InMemoryModel


# In[26]:


print(features_list)


# ### Flujo de trabajo : Objeto de interpretación > Modelo en memoria > Interpretación

# In[27]:


interpreter = Interpretation(training_data=X_train, feature_names=features_list)
im_model = InMemoryModel(xgb.predict_proba, examples=X_test, target_names=['1', '2', '3', '4'])


# In[28]:


plots = interpreter.feature_importance.plot_feature_importance(im_model, ascending=False, progressbar=False)


# In[43]:


r = interpreter.partial_dependence.plot_partial_dependence(['State'], im_model, grid_resolution=50, grid_range=(0,1), n_samples=1000, with_variance=True, figsize = (6, 4), n_jobs=-1)
yl = r[0][1].set_ylim(0, 1)


# ### Gráfico PD bivariante que muestra las interacciones entre las características "Estado" y "Temperatura" y su efecto en el "Resultado".

# In[44]:


get_ipython().run_cell_magic('time', '', "\nplots_list = interpreter.partial_dependence.plot_partial_dependence([('State', 'Temperature(F)')], im_model, grid_range=(0,1), n_samples=1000, figsize=(16, 6), grid_resolution=100, progressbar=True, n_jobs=-1)")


# ## Local Interpretable Model-Agnostic Explanations (LIME)
# 

# In[29]:


predictions = xgb_array.predict_proba(X_test.values)


# In[30]:


from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer

exp = LimeTabularExplainer(X_test.values, feature_names=features_list, discretize_continuous=True, class_names=['1', '2', '3', '4'])


# In[37]:


accidentes_nb = 600
print('Reference:', y_test.iloc[accidentes_nb])
print('Predicted:', predictions[accidentes_nb])
exp.explain_instance(X_test.iloc[accidentes_nb].values, xgb_array.predict_proba).show_in_notebook()


# In[32]:


accidentes_nb = 999
print('Reference:', y_test.iloc[accidentes_nb])
print('Predicted:', predictions[accidentes_nb])
exp.explain_instance(X_test.iloc[accidentes_nb].values, xgb_array.predict_proba).show_in_notebook()


# ## Interpretación del modelo con SHAP

# In[38]:


import shap
shap.initjs()


# In[39]:


get_ipython().run_cell_magic('time', '', '\nexplainer = shap.TreeExplainer(xgb)\nshap_values = explainer.shap_values(X_test)')


# In[76]:


X_shap = pd.DataFrame(shap_values)
X_shap.tail()


# In[95]:


print('Expected Value: ', explainer.expected_value)


# ### Importancia de las características con SHAP
# 

# In[56]:


shap.summary_plot(shap_values, X_test, plot_type="bar", color='red')


# In[59]:


shap.force_plot(explainer.expected_value, shap_values[1,:], X_test.iloc[1,:])


# In[61]:


shap.force_plot(explainer.expected_value, shap_values[3,:], X_test.iloc[3,:])


# In[62]:


shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_test.iloc[:1000,:])


# In[60]:


shap.summary_plot(shap_values, X_test)


# In[63]:


shap.dependence_plot(ind='Temperature(F)', interaction_index='State',
                     shap_values=shap_values, 
                     features=X_test,  
                     display_features=X_test)


# ## FairML
# 

# In[64]:


from fairml import audit_model
from fairml import plot_dependencies


# In[65]:


get_ipython().run_cell_magic('time', '', "\nxgb_fair = XGBClassifier(objective='binary:logistic', random_state=33, n_jobs=-1)\n\nxgb_fair.fit(X_train.values, y_train, eval_metric = 'logloss')")


# In[66]:


get_ipython().run_cell_magic('time', '', "\nfeat_importances, _ = audit_model(xgb_fair.predict, X_train, distance_metric='accuracy', direct_input_pertubation_strategy='constant-zero', number_of_runs=10, include_interactions=True)\n\nprint(feat_importances)")


# In[101]:


fig = plot_dependencies(
    feat_importances.median(),
    reverse_values=False,
    title="FairML feature dependence XGB model",
    fig_size=(8,3)
    )

