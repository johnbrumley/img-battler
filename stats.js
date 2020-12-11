let net;
let model;
const scale = 10;
const video = document.getElementById("video")

// webcams
navigator.mediaDevices
  .getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: 600,
      height: 500
    }
  })
  .then(stream => {
    video.srcObject = stream
    video.onloadedmetadata = () => {
      video.play()
    }
  })


// adding or removing from this list will add/remove the number of values
// we get from mobilnet
// const statNames = ['Beauty','Sourness','Stubble','Flakiness','Courage','Glow','literacy'];
const statNames = ['HP','MP','STR','DEX','CON','INT','WIS','CHA','LCK'];

async function generateStats(){
    if (video) {
        const results = await model.predict(net.infer(video, true));
        const data = await results.data();
        const stats = document.getElementsByClassName('stats')[0];
        // clear existing stats
        stats.innerHTML = '';
        for (let i = 0; i < data.length; i++) stats.innerHTML = stats.innerHTML + ('<p>' + statNames[i] + ': ' + Math.round(data[i]*scale) + '</p>');
    }
    requestAnimationFrame(generateStats);
}


async function loadModel() {
    
    // just grabbed a random thing that said PCA could be a single layer network ,,
    // probably we should train some kind of autoencoder

    model = tf.sequential();
    // To simulate PCA we use 1 hidden layer with a linear (relu) activation
    const encoder = tf.layers.dense({
    units: statNames.length, 
    batchInputShape:[1,256], // match size of embeds that we get from mobilnet
    activation: 'relu',
    kernelInitializer:tf.initializers.randomNormal({'seed':42}), // setting a seed gets consistent vals, might need to adjust so some stats don't always come out low/high
    // kernelInitializer:tf.initializers.constant({'value':0.5}), 
    // kernelInitializer:tf.initializers.glorotNormal({'seed':187}),
    // kernelInitializer:tf.initializers.randomUniform({'seed':42}),
    biasInitializer:"ones"});
    // const decoder = tf.layers.dense({units: 256, activation: 'relu'});

    model.add(encoder);
    // model.add(decoder); 
    await model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

    console.log('Loading mobilenet..');

    // Load the model.
    const version = 1;
    const alpha = 0.25; // less output embeds
    net = await mobilenet.load({version, alpha});
    console.log('Successfully loaded model');

    // try generating stats
    generateStats();
}

loadModel();