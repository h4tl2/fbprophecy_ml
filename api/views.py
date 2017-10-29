from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import status
from rest_framework.views import APIView

from rest_framework.response import Response
from api.serializers import UserSerializer, GroupSerializer
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer

class linearRegressionPredict(APIView):
    def get(self, request, *args, **kw):
        result = {"status" : 200, "type" : "get"}
        response = Response(result, status.HTTP_200_OK)
        return response
    
    def post(self, request, *args, **kw):
        
        df3 = pd.read_csv('df3.csv')
        features = ['B365H', 'B365D', 'B365A', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'BbAvH', 'BbAvD', 'BbAvA']
        X = df3[features]
        y = df3['FTR']
        
        tree = DecisionTreeClassifier(criterion="gini")
        tree.fit(X, y)

        print(request.data)
        F = pd.Series(request.data,index=request.data.keys())
        G = []
        for key in X.keys():
            G.append(float(F[key]))
        Res = np.array([G])
        print(Res)
        predictions = tree.predict(Res)
        result = {"status" : 200, "type" : "post", "prediction" : predictions}
        response = Response(result, status.HTTP_200_OK)
        return response
