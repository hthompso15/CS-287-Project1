'''
filename = "input.txt"
[COMPUTATIONS]
if [CONDITIONS]:
print("good")
else:
print("bad")
'''

# Import data (needs to be robust to JSON & CSV?

import string
import json

filename = "input.txt"
#message = 'tell me a pizza'
#action = 'pizza'

# Bad Data Type 1:
# Mislabeled, easily detected

WEATHER_KEYWORDS = [
    'rain',
    'snow',
    'sunshine',
    'cloud',
]

JOKE_KEYWORDS = [
    'joke',
    'funny',
    'laugh'
]

TIME_KEYWORDS = [
    'time',
    'hour',
    'minute',
    'second'
]

PIZZA_KEYWORDS = [
    'pizza',
    'pepperoni',
    'cheese'
]

# I skipped GREET since lots of messages are like "Hi Jarvis, order me pizza"
keywords = {
    'weather': WEATHER_KEYWORDS,
    'time': TIME_KEYWORDS,
    'pizza': PIZZA_KEYWORDS,
    'joke': JOKE_KEYWORDS
}
# Keyword checking logic
# If another action's keywords are in the message, data is bad\

def checkBad(message,action):
    bad = False
    for keyword in keywords.keys():
         if keyword != action.lower():
            bad_keyword = (1 in [word in message for word in keywords[keyword]])
            if bad_keyword:
                bad = True

    # Other myriad checks here
    if action not in keywords.keys():
        bad = True

    # Ascii check:
    try:
        message.encode('ascii')
    except:
        bad = True

    if not bad:
        print("good")
    else:
        print("bad")

f = filename
infile = open(f,'r')
try:
    for line in infile:
        tempDict = json.loads(line)
        checkBad(tempDict["TXT"],tempDict["ACTION"])
except:
    lines = infile.readlines()
    for i in range(0,len(lines)):
        line = lines[i]
        if (line[0] == "\""):
            line = line.split('\",')
        else:
            line = line.split(',')
        checkBad(line[0],line[1])
