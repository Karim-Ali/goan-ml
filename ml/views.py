from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import pickle
import pandas as pd
import numpy as np


# Create your views here.


@api_view(["POST"])
#my fun
def recommend(data):

    try:
        jsonobj = json.loads(data.body)
        
        keys1 = ['item_id']
        keys2 = jsonobj.keys()
        keys2 = list(keys2)
        
        if keys1 != keys2:
            raise ValueError("keys are Not permitted")
            
        item_id = jsonobj["item_id"]
        
        if not isinstance(item_id, int):
            raise ValueError("Error value sent")
        #recommender = Recommender()
        #response = recommender.recommend(item_id)
        
        ratings = pickle.load(open("ml/models/ratings.sav", 'rb'))
        ratemat = pickle.load(open("ml/models/ratemat.sav", 'rb'))
        
        item_user_ratings = ratemat[item_id]
        similar_to_item = ratemat.corrwith(item_user_ratings)
        item_corr = pd.DataFrame(similar_to_item,columns=['Correlation'])
        item_corr = item_corr.join(ratings['num of ratings'])
        pridicted = item_corr[item_corr['num of ratings']>100]
        pridicted.sort_values('Correlation',ascending=False)
        
        pridicted = pd.DataFrame.to_json(pridicted)
        return JsonResponse(pridicted,safe=False)

    except ValueError as e:

        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

@api_view(["POST"])
def predictCan(data):

    try:
        jsonobj = json.loads(data.body)
        
        keys1 = ['LeadTime','IsRepeatedGuest','PreviousCancellations','PreviousBookingsNotCanceled', 'avgPrice', 'fieldType']
        keys2 = jsonobj.keys()
        keys2 = list(keys2)
        
        if keys1 != keys2:
            raise ValueError("keys are Not permitted")
            
        transaction = []
        
        for element in keys2:
            transaction.append(jsonobj[element])
            
        for i in range(5):
            if not isinstance(transaction[i], int):
                raise ValueError("Error value sent")
        if not isinstance(transaction[5], str) :
            raise ValueError("Error value sent")
        
        #pridictor = Cancelation_Pridictor()
        #response = pridictor.pridict(transaction)
        #model = pickle.load(open("ml/models/cancelation_pridictor_model.sav", 'rb'))
        #transaction[5] = (ord(transaction[5])-64)
        #x = np.column_stack((transaction))
        x = transaction[0]+transaction[1]+transaction[2]+transaction[3]+transaction[4]
        #result = model.predict(x)
        #result = result.tolist()
        pridicted = {"is_canceled" :(x%2)}
        #pridicted = json.dumps(pridicted)
        
        return JsonResponse(pridicted,safe=False)

    except ValueError as e:

        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
    
    
@api_view(["POST"])

def updateRecommender(data):
    
    try:
        jsonobj = json.loads(data.body)
        keys1 = ['user_id', 'item_id', 'rating']
        keys2 = jsonobj.keys()
        keys2 = list(keys2)
        
        if keys1 != keys2:
            raise ValueError("keys are Not permitted")
        
        #recommender = Recommender()
        #recommender.updateModel(jsonobj)
        dataframe = pd.read_json(jsonobj)
        ratings = pd.DataFrame(dataframe.groupby('item_id')['rating'].mean())
        ratings['num of ratings'] = pd.DataFrame(dataframe.groupby('item_id')['rating'].count())
        
        ratemat = dataframe.pivot_table(index='user_id',columns='item_id',values='rating')

        pickle.dump(ratings, open("ml/models/ratings.sav", 'wb'))
        pickle.dump(ratemat, open("ml/models/ratemat.sav", 'wb'))
        return JsonResponse("model updated successfully",safe=False)

    except ValueError as e:

        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

@api_view(["POST"])

def updateCanPredictor(data):

    try:
        jsonobj = json.loads(data.body)
        keys1 = ['LeadTime','IsRepeatedGuest','PreviousCancellations','PreviousBookingsNotCanceled', 'fieldType', 'avgPrice']
        keys2 = jsonobj.keys()
        keys2 = list(keys2)
        
        if keys1 != keys2:
            raise ValueError("keys are Not permitted")
        #pridictor = Cancelation_Pridictor()
        #pridictor.updateModel(jsonobj)

        dataframe = pd.read_json(jsonobj)

        from sklearn.model_selection import train_test_split
        #column_names = ['LeadTime','IsRepeatedGuest','PreviousCancellations','PreviousBookingsNotCanceled', 'avgPrice', 'fieldType']
        leadtime = dataframe['LeadTime']
        isrepeatedguest = dataframe['IsRepeatedGuest'] 
        previouscancellations = dataframe['PreviousCancellations']
        previousbookingsnotcanceled = dataframe['PreviousBookingsNotCanceled']
        adr = dataframe['avgPrice']
        category=dataframe.fieldType.astype("category").cat.codes
        category=pd.Series(category)
        
        y = dataframe['IsCanceled']
        x = np.column_stack((leadtime,isrepeatedguest,previouscancellations,previousbookingsnotcanceled,adr,category))
        
        x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=0)
        
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(learning_rate=0.075,
                                      max_depth = 5, 
                                      n_estimators = 1000,
                                      scale_pos_weight=3)
        xgb_model.fit(x_train, y_train)
        pickle.dump(xgb_model, open("ml/models/cancelation_pridictor_model.sav", 'wb'))
        
        return JsonResponse("model updated successfully",safe=False)

    except ValueError as e:

        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
