## hereditary path           <!-- name of the story - just for debugging -->
* greet
  - utter_greet
* consult_hereditary       <!-- user utterance, in the following format: * intent{"entity_name": value} -->
  - utter_hereditary
  - utter_did_that_help
* affirm
  - utter_happy

## hereditary path 2          <!-- name of the story - just for debugging -->
* greet
  - utter_greet
* consult_hereditary       <!-- user utterance, in the following format: * intent{"entity_name": value} -->
  - utter_hereditary
  - utter_did_that_help
* deny
  - utter_goodbye

## symptomn path 1               <!-- this is already the start of the next story -->
* greet
  - utter_greet             <!-- action of the bot to execute -->
* consult_symptom 
  - utter_symptom_comun
  - utter_did_that_help
* affirm
  - utter_happy

## symptomn path 2
* greet
  - utter_greet
* consult_symptom 
  - utter_symptom_comun
  - utter_did_that_help
* deny
  - utter_goodbye

## say goodbye
* goodbye
  - utter_goodbye

## bot challenge
* bot_challenge
  - utter_iamabot
