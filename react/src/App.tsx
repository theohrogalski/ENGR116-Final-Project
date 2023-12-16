import React, {useState} from 'react';
import './App.css';
import { FileUploader } from "react-drag-drop-files";
import * as tf from '@tensorflow/tfjs';
import Resizer from "react-image-file-resizer";

const fileTypes = ["JPG", "PNG", "GIF"];

//Class names in same order as the model was trained
const classNames = ["daisy", "dandelion", "rose", "sunflower","tulip"]

//Resize image with program-specific defaults
const resizeFile = (file: File) =>
  new Promise((resolve) => {
    Resizer.imageFileResizer(
      file,
      180,
      180,
      "JPEG",
      100,
      0,
      (uri) => {
        resolve(uri);
      },
      "base64",
      180,
      180
    );
  });

function App() {
  const [file, setFile] = useState<string | null>(null);
  const [type, setType] = useState<string>("image");
  const [confidence, setConfidence] = useState<number>(100);

  const handleChange = async (file: File) => {
    //Create temp image to calculate prediction from
    const imageData = await resizeFile(file);

    const socket = new WebSocket("ws://localhost:8080/websocket");
    
    //Log active socket (it worked!)
    
    //Process image into tensor after socket is opened and then pass to server to run model against
    socket.addEventListener("open", event => {
      console.log('Opened socket with procesing service.');

      //Create new image object to temporarily hold image data and let tensorflow read from
      let imageObj = new Image();
      
      //Run tensor calculations
      //Has to wait for DOM to load the image from the given source below before using it to calculate
      //May run instantanously, may run after a long period of being haulted, depends on how the browser is implemented
      imageObj.onload = async () => {
        console.log(imageObj);

        //Create tensor from image
        //Tensor dimentions - (width, height, channels) -> (180, 180, 3)
        let tensorImage = tf.browser.fromPixels(imageObj);
        //Expand tensor from (180, 180, 3) to (1, 180, 180, 3)
        let expandedTensor = tf.expandDims(tensorImage);

        //Relay input tensor to processing server (express host)
        socket.send(JSON.stringify(await expandedTensor.array()));

        //Let garbage collector know to collect finished tensors (Memory Optimization)
        tf.dispose(expandedTensor);
        tf.dispose(tensorImage);

      }
    
      //Set the image source as the resized image data
      imageObj.src = imageData as string;

      //Force size (Tensorflow complains if this isn't done)
      imageObj.height = 180;
      imageObj.width = 180;
    });

    //Take received result from server and translate into human readability
    socket.addEventListener("message", async (event) => {
      console.log(event.data)

      let result = tf.tensor(JSON.parse(event.data));

      //Debug result outputs
      result.print();

      //Calculate which axis has the highest result (most confidance)
      //Strip first dimention in returned 2D array to check against 
      let resultArr = await tf.argMax(result, 1).data()
      let num = resultArr[0];

      //Find prediction by selecting the name at which the axis was highest
      let prediction = classNames[num];
      console.log(prediction);
      setType(prediction);

      //Find confidance from result in whole tensor array
      //Confidance is reported on a log scale, and as such is hard to interpret. More or less higher number good when result is correct.
      let totalConfidenceArray = await result.data();
      setConfidence(totalConfidenceArray[num]);

      //Memory optimizations (Don't leave garbage on the heap!)
      socket.close();
      tf.dispose(result);
    });

    //Pass resized image to react display
    setFile(imageData as string);
  };
  
  //Generate React frame
  return (
    <div className="App">
      <header className="App-header">
        <h1>Flower Image Classification:</h1>

        {file ? 
          <div>
            <img src={file}/>
            <p>This is a {type}! Confidance: {confidence}</p>
          </div> 
        : 
          <p>No Image uploaded!</p>
        }

        <h2>Drag Photo Below:</h2>

        <FileUploader handleChange={handleChange} name="file" types={fileTypes} />

        <p>Supported File Types: JPEG, PNG, GIF</p>
        <p>By Jack Gonser & Theo Rogalski 2023</p>
      </header>
    </div>
  );
}

export default App;
