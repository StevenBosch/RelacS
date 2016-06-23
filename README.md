Running Relacs
===================
To run Relacs you first need to install the requirements (as usual):
`pip install -r requirements.txt`
First you need to make sure that a database has been set up:
```
python site/manage.py migrate
```
You can run the debug server using the following command:
```
python site/manage.py runserver
```
Then you can go to http://localhost:8000/ to visit the website.

Next you can process a new .hdf5 file:
```
python code/process_file.py _.hdf5
```
Finally, you can update the server with the newly processed file using:
```
python site/manage.py addrecording_new_.wav Windowsprediction.pickle
````

