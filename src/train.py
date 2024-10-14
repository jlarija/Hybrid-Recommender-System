import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix # for constructing sparse matrix
from lightfm import LightFM # for model
from lightfm.evaluation import auc_score
import time
import sklearn
from sklearn import model_selection
import joblib

class PrepData(object):
    "class to prepare the data for the Recommender Engine"
    def __init__(self, path=None):
        #orders data
        self.order_df = pd.read_excel(path,'order')
        #customers data
        self.customer_df = pd.read_excel(path,'customer')
        #products data
        self.product_df = pd.read_excel(path,'product')
        self.merged_df = pd.merge(self.order_df,self.customer_df,left_on=['CustomerID'], right_on=['CustomerID'], how='left')
        self.merged_df = pd.merge(self.merged_df,self.product_df,left_on=['StockCode'], right_on=['StockCode'], how='left')

    def unique_users(self,data, column):
        # find the list of unique users for the label encoder
        return np.sort(data[column].unique())
    
    def unique_items(self, data, column):
        # find the list of unique items to encode labels
        item_list = data[column].unique()
        return item_list
    
    def features_to_add(self, customer, column1,column2,column3):
        customer1 = customer[column1]
        customer2 = customer[column2]
        customer3 = customer[column3]
        return pd.concat([customer1,customer3,customer2], ignore_index = True).unique()
        
    def mapping(self, user_list, item_list, feature_unique_list):
    #creating empty output dicts
        user_to_index_mapping = {}
        index_to_user_mapping = {}
        # Create id mappings to convert user_id
        for user_index, user_id in enumerate(user_list):
            user_to_index_mapping[user_id] = user_index
            index_to_user_mapping[user_index] = user_id
        item_to_index_mapping = {}
        index_to_item_mapping = {}
        # Create id mappings to convert item_id
        for item_index, item_id in enumerate(item_list):
            item_to_index_mapping[item_id] = item_index
            index_to_item_mapping[item_index] = item_id
        feature_to_index_mapping = {}
        index_to_feature_mapping = {}
        # Create id mappings to convert feature_id
        for feature_index, feature_id in enumerate(feature_unique_list):
            feature_to_index_mapping[feature_id] = feature_index
            index_to_feature_mapping[feature_index] = feature_id
        return user_to_index_mapping, index_to_user_mapping, item_to_index_mapping, index_to_item_mapping, feature_to_index_mapping, index_to_feature_mapping
    
    def interactions(self, data, row, col, value, row_map, col_map):
        #converting the row with its given mappings
        row = data[row].apply(lambda x: row_map[x]).values
        #converting the col with its given mappings
        col = data[col].apply(lambda x: col_map[x]).values
        value = data[value].values
        #returning the interaction matrix
        return coo_matrix((value, (row, col)), shape = (len(row_map), len(col_map)))

    def prep_data(self):
        # model has been separately validate, and therefore no train/test split is made 
        user_list = self.unique_users(self.order_df, "CustomerID")
        item_list = self.unique_items(self.product_df, "Product Name")
        feature_unique_list = self.features_to_add(self.customer_df,'Customer Segment',"Age","Gender")
        user_to_index_mapping, index_to_user_mapping, item_to_index_mapping, index_to_item_mapping, \
        feature_to_index_mapping, index_to_feature_mapping = self.mapping(user_list, item_list, feature_unique_list)

        user_to_product = self.merged_df[['CustomerID','Product Name','Quantity']]
        #Calculating the total quantity(sum) per customer-product
        user_to_product = user_to_product.groupby(['CustomerID','Product Name']).agg({'Quantity':'sum'}).reset_index()

        product_to_feature = self.merged_df[['Product Name','Customer Segment','Quantity']]
        #Calculating the total quantity(sum) per customer_segment-product
        product_to_feature = product_to_feature.groupby(['Product Name','Customer Segment']).agg({'Quantity':'sum'}).reset_index()
        user_to_product_interaction = self.interactions(user_to_product, "CustomerID", "Product Name", "Quantity", user_to_index_mapping, item_to_index_mapping)
        product_to_feature_interaction = self.interactions(product_to_feature, "Product Name", "Customer Segment","Quantity",item_to_index_mapping, feature_to_index_mapping)
        # Save the user-to-product interaction matrix
        joblib.dump(user_to_product_interaction, 'model/user_to_product_interaction.pkl')
        joblib.dump(item_list, 'model/item_list.pkl')
        joblib.dump(user_to_index_mapping, 'model/user_to_index_mapping.pkl')
        joblib.dump(product_to_feature_interaction, 'model/product_to_feature.pkl')
        return user_to_product_interaction, product_to_feature_interaction
    

class RecommenderEngine():
    def __init__(self):
        input_data = PrepData(path='data/Rec_sys_data.xlsx')
        self.user_to_product_interaction, self.product_to_feature_interaction = input_data.prep_data()

    def train_model(self):
        model_with_features = LightFM(loss = "logistic")
        start = time.time()
        #===================
        # fitting the model with hybrid collaborative filtering + content based (product + features)
        model_with_features.fit_partial(self.user_to_product_interaction,
        user_features=None,
        item_features=self.product_to_feature_interaction,
        sample_weight=None,
        epochs=10,
        num_threads=20,
        verbose=False)
        #===================
        end = time.time()
        print("time taken = {0:.{1}f} seconds".format(end - start, 2))
        # Save the model to a file
        joblib.dump(model_with_features, 'model/trained_model.pkl')
        print("Model saved successfully.")
        return model_with_features
    
if __name__ == '__main__':
    RecommenderEngine().train_model()