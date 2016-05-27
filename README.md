Running the website
===================
To run the website you first need to install the requirements (as usual):
`pip install -r requirements.txt`
Then you can cd into the site directory. First you need to make sure that a
database has been set up:
```
python manage.py migrate
```
Next you can load an example database, there is no script to deal with trained
data yet :(
```
python manage.py loaddata example.json
```
Finally you need to run the debug server using the following command:
```
python manage.py runserver
````
Then you can go to http://localhost:8000/ to visit the website.
