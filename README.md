# ENGR116-Final-Project

Train & Deploy a Tensorflow neural network

Utilizes a React front-end being served by an express.js server.

Model is trained via a python script, and pictures are processed server side using websockets between the browser and the express.js server.

## To train model:
> python3 ImprovDrop.py

## To run python test script
Place in image so that Images/serve_image.jpeg exists
> python3 testingsdf.py

## To deploy web server
Starting from top directory of project:
> cd react && yarn build && cd ../express-runtime && yarn start

Jack Gonser & Theodore Rogalski
