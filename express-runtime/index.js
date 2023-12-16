import express from 'express';
import path from 'path';
import expressWS from "express-ws";

import * as tf from '@tensorflow/tfjs-node';

//Create express service
const app = express();

//Load model from local storage
const handler = tf.io.fileSystem("./js_model/model.json");
const model = await tf.loadLayersModel(handler);

//Enable expressWS (Runs websocket server under exisiting express HTTPServer)
let ws = expressWS(app);

//Register websocet path
app.ws('/websocket', function(ws, req) {

    //Take input tensor and calculate result from given model
    ws.on('message', async function(msg) {
        //Create tensor from incoming socket message
        let data = JSON.parse(msg);
        let tensor = tf.tensor(data);

        //Run model off of given data
        let result = model?.predict(tensor);
        console.log("Outgoing prediction: ");
        result.print();

        //GC input
        tf.dispose(tensor);

        if (result instanceof tf.Tensor) {
            //Serialize and transport result back to web interface
            ws.send(JSON.stringify(await result.array()));

            //GC output
            tf.dispose(result);
        } else {
            //Error handling
            ws.send('Error!');
            console.log("ERR: Didn't calculate a resulting tensor from neural net! Input:");
            console.log(data);
        }

    });
});

//Serve the react build statically (like traditional web server)
app.use(
    express.static(
        path.join(
            process.cwd(), '../react/build')
            )
    );

//Hi!
app.use((req, res) => {
    res.status(200).send('Hello, world!');
});


// Start the server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
    console.log(`App listening on port ${PORT}`);
    console.log('Press Ctrl+C to quit.');
});

