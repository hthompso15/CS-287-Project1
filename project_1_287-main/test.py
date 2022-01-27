
import pickle

brain = pickle.load(open("jarvis_WIRELESSOCTOPUS.pkl", 'rb'))
result = brain.predict(["what is the temperature"])
print(result)

