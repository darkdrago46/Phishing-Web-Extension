# Phishing-Web-Extension

A web extension tool that is used to detect phishing URLs and alert the user if needed. It uses machine learning model with logistic regression to accomplish its task.

The main objectives are :

Read the URL from the browser tab
Store it in a variable and then send it to an intermediatory server using GET/POST method
The intermediatory server in-turn calls the python model passing the URL as argument.
The Model tests the URL and returns the predicted result to the server which then forwards it to the user.
If the URL is not secure then display an alert window.
