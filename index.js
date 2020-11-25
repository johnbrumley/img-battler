let net;
const scale = 10;

// adding or removing from this list will add/remove the number of values
// we get from mobilnet
const statNames = ['Beauty','Sourness','Stubble','Flakiness','Courage','Glow','literacy'];


async function app() {

    // just grabbed a random thing that said PCA could be a single layer network ,,
    // probably we should train some kind of autoencoder

    const model = tf.sequential();
    // To simulate PCA we use 1 hidden layer with a linear (relu) activation
    const encoder = tf.layers.dense({
      units: statNames.length, 
      batchInputShape:[1,256], // match size of embeds that we get from mobilnet
      activation: 'relu',
      kernelInitializer:tf.initializers.randomNormal({'seed':2234}), // setting a seed gets consistent vals
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

    // Make a prediction through the model on our image.
    const img1 = document.getElementById('img1');
    const img2 = document.getElementById('img2');

    const r1 = model.predict(net.infer(img1, true));
    const r2 = model.predict(net.infer(img2, true));
    d1 = await r1.data();
    d2 = await r2.data();

    const p1 = document.getElementsByClassName('p1')[0];
    const p2 = document.getElementsByClassName('p2')[0];

    for (let i = 0; i < d1.length; i++) p1.innerHTML = p1.innerHTML + ('<p>' + statNames[i] + ': ' + Math.round(d1[i]*scale) + '</p>');
    for (let i = 0; i < d2.length; i++) p2.innerHTML = p2.innerHTML + ('<p>' + statNames[i] + ': ' + Math.round(d2[i]*scale) + '</p>');
}

app();