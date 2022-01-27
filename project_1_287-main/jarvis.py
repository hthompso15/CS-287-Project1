# Jarvis

# Import packages
import websocket
import requests
import sqlite3
from sqlite3 import Error
from os.path import exists
from botsettings import API_TOKEN, APP_TOKEN

# new to project 02
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import pickle

# external data imports
import os
import string
from bs4 import BeautifulSoup
import re
import json

# globals are removed for project02
# # global variables
# inTraining = None
# hasAction = None
# action = ""

# Section to create database file
class DataBase:

    def __init__(self, database_path):
        self.connection = self.sql_connection(name=database_path)

        # Check if table exists, if not create it
        # Ripped from https://pythonexamples.org/python-sqlite3-check-if-table-exists/
        c = self.connection.cursor()

        # get the count of tables with the name
        c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='training_data' ''')

        # if the count is 1, then table exists
        if c.fetchone()[0] == 1:
            print('Table exists.')
        else:
            self.create_table()

        # Old table creation code
        #if (not(exists(database_path))):
        #    self.create_table()

    def load_external(self,directory):
        for file in os.listdir(directory):
            f = os.path.join(directory,file)
            infile = open(f,'r')
            try:
                for line in infile:
                    tempDict = json.loads(line)
                    self.insert_training(tempDict["TXT"],tempDict["ACTION"])
            except:
                arr = []
                lines = infile.readlines()
                for i in range(0,len(lines)):
                    line = lines[i]
                    if(line[0] == "\""):
                        line = line.split('\",')
                    else:
                        line = line.split(',')
                    arr.append([line[0],line[1][0:-1]])
                for vals in arr:
                    self.insert_training(vals[0], vals[1])

            infile.close()
        print("done with external data")

    def sql_connection(self, name='jarvis.db'):
        try:
            connection = sqlite3.connect(name, check_same_thread=False)
            print('database connected')
            return connection
        except Error:
            print(Error)

    def create_table(self):
        cursorObj = self.connection.cursor()
        try:
            cursorObj.execute("CREATE TABLE training_data( message TEXT, action TEXT)")
            self.connection.commit()

        except Error:
            print('create_table:', Error)

    def delete_all_tasks(self):
        sql = 'DELETE FROM training_data'
        cur = self.connection.cursor()
        cur.execute(sql)
        self.connection.commit()

    def insert_training(self, message, action):
        sql = 'INSERT INTO training_data (message,action) VALUES (?,?)'
        cur = self.connection.cursor()
        cur.execute(sql, (message, action))
        #print("Inserted")
        self.connection.commit()


##End Database Section

class Jarvis:
    def __init__(self,
                 database_path='jarvis.db',
                 pickled_model_path='jarvis_WIRELESSOCTOPUS.pkl'):
        self.action = None
        self.hasAction = False
        self.inTraining = False
        self.inTesting = False
        self.searching = False
        self.local_inTraining = None
        self.local_inTesting = None
        self.local_hasAction = None
        self.local_searching = None
        self.database_path = database_path
        self.pickled_model_path = pickled_model_path

        # Setup database connection
        self.DB = DataBase(database_path)

        #self.DB.load_external('data')

        self.classifier = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC())
        ])
        # establish websocket connection
        self.ws = websocket.WebSocketApp(url_string,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        self.ws.on_open = self.on_open


    #def predict(self, data):
    #    results = self.model.predict(data)
    #    return results


    def run(self):
        self.ws.run_forever()

    # Added dummy parameter to avoid callback
    def on_open(self, _=None):
        print("###open###")

    def on_error(self, error):
        print(error)

    def on_close(self):
        print("### closed ###")

    def on_message(self, burner, message):
        # a simple acknowledgement requires by slack
        message = json.loads(message)

        envelope_id = message['envelope_id']  # this line is giving callback error
        resp = {'envelope_id': envelope_id}
        self.ws.send(str.encode(json.dumps(resp)))

        # print flags info (DEBUG)
        print("inTraining: ", self.inTraining)
        print("inTesting: ", self.inTesting)
        print("hasAction: ", self.hasAction)

        print("Websocket: ", self.ws)
        print('"type": {}, "retry_attempt: {}'.format(message['payload']['event']['text'], message['retry_attempt']))

        if message['payload']['event'].__contains__("client_msg_id"):
            message_received = message['payload']['event']['text']
            channel = message['payload']['event']['channel']

            # Conversational logic
            # Entering training mode
            if (message_received == "training time") and (not self.inTraining) and (not self.inTesting)and (not self.searching):
                message_to_send = "OK, I'm ready for training. What NAME should this ACTION be?"
                send_message_from_jarvis(channel, message_to_send)
                self.local_inTraining = True
                self.local_inTesting = False
                self.local_hasAction = False

            # Entering testing mode
            elif (message_received == "testing time") and (not self.inTraining)and (not self.inTesting)and (not self.searching):
                message_to_send = "Let me prepare my brain quickly... \n (ʃ⌣́,⌣́ƪ)"
                send_message_from_jarvis(channel, message_to_send)
                self.prepare_model()
                message_to_send = "OK, I'm ready for testing! What command should I classify? ᕙ(´෴`)ᕗ"
                send_message_from_jarvis(channel, message_to_send)
                self.local_inTesting = True

            elif (message_received == "search time") and (not self.inTraining)and (not self.inTesting)and (not self.searching):
                message_to_send = "OK, what should I search?"
                send_message_from_jarvis(channel, message_to_send)
                self.local_searching = True

            # Training Process
            if (self.inTraining):
                if not self.hasAction:
                    self.action = message_received
                    self.action = self.action.upper()
                    message_to_send = "Ok, let's call this action `{}`. Now give me some training text".format(self.action)
                    send_message_from_jarvis(channel, message_to_send)
                    self.local_inTraining = True
                    self.local_inTesting = False
                    self.local_hasAction = True
                    self.local_searching = False

                elif self.hasAction:
                    if (message_received == "done"):
                        message_to_send = "OK, I'm finished training"
                        send_message_from_jarvis(channel, message_to_send)
                        self.local_inTraining = False
                        self.local_inTesting = False
                        self.local_hasAction = False
                        self.local_searching = False

                    else:
                        message_to_send = "OK, I've got it! what else?"
                        send_message_from_jarvis(channel, message_to_send)
                        text2db = message_received
                        print("action: " + self.action)
                        self.DB.insert_training(text2db, self.action)  # callback error -- no such table: training_data
                        self.local_inTraining = True
                        self.local_inTesting = False
                        self.local_hasAction = True
                        self.local_searching = False

            # Testing Process
            elif (self.inTesting):
                if (message_received == "done"):
                    message_to_send = "OK, I'm finished testing"
                    send_message_from_jarvis(channel, message_to_send)
                    self.local_inTraining = False
                    self.local_inTesting = False
                    self.local_hasAction = False
                    self.local_searching = False

                else:
                    message_to_send = "Hm... let me think (ー_ーゞ"
                    send_message_from_jarvis(channel, message_to_send)
                    print("Classifying now:")

                    prediction = self.classifier.predict([message_received])
                    print(message_received,'->',prediction)

                    message_to_send = "I think you're asking me to %s! \n What else should I try?"%prediction[0]
                    send_message_from_jarvis(channel, message_to_send)

                    #self.DB.insert_training(text2db, self.action)  # callback error -- no such table: training_data
                    self.local_inTraining = False
                    self.local_inTesting = True
                    self.local_hasAction = False
                    self.local_searching = False

            # Searching
            elif (self.searching):
                if (message_received == "done"):
                    message_to_send = "OK, no more searching!"
                    send_message_from_jarvis(channel, message_to_send)
                    self.local_inTraining = False
                    self.local_inTesting = False
                    self.local_hasAction = False
                    self.local_searching = False
                elif (message_received == "help"):
                    message_to_send = "SEARCH MODE:\n"+("-"*20)+'\nWhile in Search Mode, send Jarvis a search term. '+ \
                                                              'Jarvis will then send you the top search result retrieved from' \
                                                                 ' Yahoo Answers.\n'+("-"*20)
                    send_message_from_jarvis(channel, message_to_send)
                else:
                    message_to_send = "Let me look that up for you on Yahoo Answers!"
                    send_message_from_jarvis(channel, message_to_send)
                    print("Searching now:")

                    results = get_search_results(message_received)
                    message_to_send = "Here's what I found:\n"+results[0]+"\n What else should I look up?"
                    send_message_from_jarvis(channel, message_to_send)

        self.inTraining = self.local_inTraining
        self.inTesting = self.local_inTesting
        self.hasAction = self.local_hasAction
        self.searching = self.local_searching

    # Preps SciKitLearn Pipeline
    # Fits Model on data in DataBase
    # Returns a trained model
    def prepare_model(self):

        # Prepare Data
        conn = self.DB.sql_connection()
        df = pd.read_sql("SELECT * FROM training_data", conn)
        X, y = df['message'], df['action']
        target_names = list(set(y.tolist()))

        # change test_size to 0.2 or something for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

        self.classifier.fit(X_train,y_train)
        predicted = self.classifier.predict(X_test)

        # classifier performance on testing data
        print(np.mean(predicted == y_test))
        # test user input
        print(self.classifier.predict(['This is a breakfast time']))

    # predict user input text
    def predict_command(self, input_data):
        return self.classifier.predict([input_data])

    # save pickle for competition 2
    def pickle_model(self):
        pickle.dump(self.classifier, open(self.pickled_model_path, 'wb'))

def send_message_from_jarvis(channel, message):
    # Use requests to send message to Slack
    res = requests.post("https://slack.com/api/chat.postMessage",
                        {'token': API_TOKEN,
                         'channel': channel,
                         'text': message}).json()

def get_search_results(search_term):
    s = '+'.join(search_term.split())
    url = 'https://search.yahoo.com/search?q=' + s
    html = requests.get(url).text

    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for div in soup.find_all('div', class_=re.compile('dd algo')):
        links.append(div.find('a').get('href'))
    return links

if __name__ == "__main__":
    # Use the “APP” token when starting the websocket connection and the “API” token when posting a message
    # Get token. retrieve url for websocket connection
    url_string = requests.post("https://slack.com/api/apps.connections.open",
                               headers={"Content-type": "application/x-www-form-urlencoded",
                                        "Authorization": "Bearer %s" % APP_TOKEN}).json()['url']

    database_path = 'jarvis.db'

    # this debug tool views verbose connection information
    websocket.enableTrace(True)

    jarvis = Jarvis(database_path)
    jarvis.run()

    # you need to stop the jarvis.py in order to get here
    # this is for testing NLP and machine learning
    jarvis.prepare_model()
    jarvis.pickle_model()

    brain = pickle.load(open("model.pkl", 'rb'))
    result = brain.predict(["Hello funny roboooot!"])
    print(result)


